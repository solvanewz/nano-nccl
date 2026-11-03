#include "cost_profile.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace nano_nccl::sim {

Table::Table(std::vector<std::pair<double, double>> points) : points_(std::move(points)) {
    if (points_.empty()) throw std::invalid_argument("Table: no points");
    for (std::size_t i = 1; i < points_.size(); ++i) {
        if (!(points_[i].first > points_[i - 1].first)) throw std::invalid_argument("Table: x not ascending");
    }
}

double Table::operator()(double x) const {
    if (points_.empty()) throw std::logic_error("Table: empty");
    if (points_.size() == 1) return points_[0].second;
    auto lerp = [x](const std::pair<double, double>& a, const std::pair<double, double>& b) {
        return a.second + (b.second - a.second) * (x - a.first) / (b.first - a.first);
    };
    if (x <= points_.front().first) return points_.front().second;  // tables start at the smallest size of interest
    const auto hi = std::lower_bound(points_.begin(), points_.end(), x,
                                     [](const std::pair<double, double>& p, double v) { return p.first < v; });
    if (hi == points_.end()) return std::max(0.0, lerp(points_[points_.size() - 2], points_.back()));
    if (hi->first == x) return hi->second;
    return lerp(*(hi - 1), *hi);
}

double CostProfile::scalar(const std::string& name) const {
    const auto it = scalars.find(name);
    if (it == scalars.end()) throw std::out_of_range("profile has no scalar '" + name + "'");
    return it->second;
}

double CostProfile::scalar_or(const std::string& name, double fallback) const {
    const auto it = scalars.find(name);
    return it == scalars.end() ? fallback : it->second;
}

const Table& CostProfile::table(const std::string& name) const {
    const auto it = tables.find(name);
    if (it == tables.end()) throw std::out_of_range("profile has no table '" + name + "'");
    return it->second;
}

const Table& CostProfile::table_for(const std::string& name, int rank) const {
    const auto it = tables.find(name + "@" + std::to_string(rank));
    return it != tables.end() ? it->second : table(name);
}

double CostProfile::scalar_for(const std::string& name, int rank, double fallback) const {
    const auto it = scalars.find(name + "@" + std::to_string(rank));
    return it != scalars.end() ? it->second : scalar_or(name, fallback);
}

CostProfile CostProfile::parse(const std::string& text, const std::string& where) {
    CostProfile p;
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    bool header = false, have_key = false;
    auto fail = [&](const std::string& what) {
        throw std::invalid_argument(where + ":" + std::to_string(lineno) + ": " + what);
    };
    while (std::getline(in, line)) {
        ++lineno;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string word;
        ls >> word;
        if (word.empty()) continue;
        if (!header) {
            int version = 0;
            if (word != "nano-ep-profile" || !(ls >> version) || version != 1) fail("expected 'nano-ep-profile 1'");
            header = true;
        } else if (word == "key") {
            p.key = exec::ProfileKey::parse(line.substr(4));
            have_key = true;
        } else if (word == "meta") {
            p.meta.push_back(line.size() > 5 ? line.substr(5) : "");
        } else if (word == "scalar") {
            std::string name;
            double v = 0;
            if (!(ls >> name >> v)) fail("bad scalar line");
            p.scalars[name] = v;
        } else if (word == "table") {
            std::string name, xu, yu;
            if (!(ls >> name >> xu >> yu)) fail("bad table line");
            std::vector<std::pair<double, double>> pts;
            bool closed = false;
            while (std::getline(in, line)) {
                ++lineno;
                if (line.empty() || line[0] == '#') continue;
                if (line == "end") {
                    closed = true;
                    break;
                }
                std::istringstream ps(line);
                double x = 0, y = 0;
                if (!(ps >> x >> y)) fail("bad table row in '" + name + "'");
                pts.push_back({x, y});
            }
            if (!closed) fail("table '" + name + "' has no 'end'");
            try {
                p.tables[name] = Table(std::move(pts));
            } catch (const std::invalid_argument& e) {
                fail("table '" + name + "': " + e.what());
            }
            p.units[name] = {xu, yu};
        } else {
            fail("unknown line '" + word + "'");
        }
    }
    if (!header) fail("empty profile");
    if (!have_key) fail("missing key line");
    return p;
}

CostProfile CostProfile::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open profile " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse(ss.str(), path);
}

std::string CostProfile::serialize() const {
    std::ostringstream o;
    o << std::setprecision(9);
    o << "nano-ep-profile 1\n";
    o << "key " << key.to_string() << "\n";
    for (const std::string& m : meta) o << "meta " << m << "\n";
    for (const auto& [name, v] : scalars) o << "scalar " << name << " " << v << "\n";
    for (const auto& [name, t] : tables) {
        const auto u = units.count(name) ? units.at(name) : std::make_pair(std::string("x"), std::string("us"));
        o << "table " << name << " " << u.first << " " << u.second << "\n";
        for (const auto& [x, y] : t.points()) o << x << " " << y << "\n";
        o << "end\n";
    }
    return o.str();
}

void CostProfile::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write profile " + path);
    f << serialize();
    if (!f) throw std::runtime_error("short write to " + path);
}

FabricRates link_rates(double gbps, double message_us, std::size_t chunk_bytes, double exchange_us) {
    if (!(gbps > 0)) throw std::invalid_argument("link_rates: gbps must be positive");
    FabricRates f;
    f.tx_us_per_mib = f.rx_us_per_mib = 1048576.0 * 8 / (gbps * 1e3);
    f.message_us = message_us;
    f.chunk_bytes = chunk_bytes;
    f.exchange_us = exchange_us;
    return f;
}

FabricRates copy_rates(const CostProfile& p, std::size_t chunk_bytes) {
    if (chunk_bytes == 0) throw std::invalid_argument("copy_rates: chunk_bytes must be positive");
    const double mib = static_cast<double>(chunk_bytes) / 1048576.0;
    FabricRates f;
    f.tx_us_per_mib = p.table(cost::kTxRemote)(static_cast<double>(chunk_bytes)) / mib;
    f.rx_us_per_mib = p.table(cost::kRxRemote)(static_cast<double>(chunk_bytes)) / mib;
    f.message_us = p.scalar(cost::kHostPost);
    f.chunk_bytes = chunk_bytes;
    f.exchange_us = p.scalar(cost::kCopyLatency) + p.scalar(cost::kHostExchangeIssue);
    return f;
}

planner::CostModel planner_cost_model(const CostProfile& p, int hidden, double compute_scale,
                                      const FabricRates& fabric) {
    auto ns = [](double us) { return static_cast<std::int64_t>(std::llround(us * 1000.0)); };
    auto table = [&](const char* name, double scale) {
        std::vector<std::pair<std::int64_t, std::int64_t>> pts;
        for (const auto& [x, y] : p.table(name).points()) {
            pts.push_back({static_cast<std::int64_t>(std::llround(x)), ns(y * scale)});
        }
        return planner::IntTable(std::move(pts));
    };
    planner::CostModel m;
    m.row_bytes = static_cast<std::int64_t>(hidden) * 2;  // bf16
    m.expert_ns = table(cost::kExpertFfn, compute_scale);
    m.permute_ns = table(cost::kPermute, compute_scale);
    m.expert_issue_ns = table(cost::kHostExpertIssue, 1.0);
    auto ps_per_byte = [](double us_per_mib) {
        return static_cast<std::int64_t>(std::llround(us_per_mib * 1e6 / 1048576.0));
    };
    m.tx_ps_per_byte = ps_per_byte(fabric.tx_us_per_mib);
    m.rx_ps_per_byte = ps_per_byte(fabric.rx_us_per_mib);
    m.message_ns = ns(fabric.message_us);
    m.chunk_bytes = static_cast<std::int64_t>(fabric.chunk_bytes);
    m.exchange_ns = ns(fabric.exchange_us);
    m.validate();
    return m;
}

const Table& schedule_table(const CostProfile& p, const std::string& scheduler, int world) {
    const std::string base = std::string(cost::kHostSchedule) + scheduler;
    const std::string at_world = base + "_w" + std::to_string(world);
    return p.tables.count(at_world) ? p.tables.at(at_world) : p.table(base);
}

}  // namespace nano_nccl::sim
