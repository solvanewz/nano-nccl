#include "exec/exec_trace.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace nano_nccl::exec {

OpKind parse_op_kind(const std::string& name) {
    for (int k = 0; k <= static_cast<int>(OpKind::kRetire); ++k) {
        if (name == op_kind_name(static_cast<OpKind>(k))) return static_cast<OpKind>(k);
    }
    throw std::invalid_argument("unknown op kind '" + name + "'");
}

std::string LayerTrace::serialize() const {
    std::ostringstream o;
    o << std::setprecision(10);
    o << "nano-ep-trace 1\n";
    o << "key " << key.to_string() << "\n";
    o << "options overlap=" << (options.overlap ? 1 : 0) << " lookahead=" << options.lookahead << " iters=" << iters
      << "\n";
    o << "expert_group";
    for (int g : groups.expert_group) o << " " << g;
    o << "\norder";
    for (int g : groups.order) o << " " << g;
    o << "\n";
    for (std::size_t r = 0; r < ranks.size(); ++r) {
        const TraceRank& t = ranks[r];
        o << "rank " << r << " tokens=" << t.tokens << " route=" << t.route << " counts=" << t.counts
          << " pipeline=" << t.pipeline << " reduce=" << t.reduce << " total=" << t.total << "\n";
        o << "counts " << r;
        for (std::int64_t n : t.expert_counts) o << " " << n;
        o << "\n";
    }
    for (const TraceOp& op : ops) {
        o << "op " << op.rank << " " << op.op << " " << op_kind_name(op.kind) << " " << op.step << " "
          << op.issue_begin << " " << op.issue_end << " " << op.complete << "\n";
    }
    return o.str();
}

LayerTrace LayerTrace::parse(const std::string& text, const std::string& where) {
    LayerTrace t;
    std::istringstream in(text);
    std::string line;
    int lineno = 0;
    bool header = false, have_key = false;
    auto fail = [&](const std::string& what) {
        throw std::invalid_argument(where + ":" + std::to_string(lineno) + ": " + what);
    };
    // Parses "name=value" into a double.
    auto field = [&](std::istringstream& ls, const std::string& name) {
        std::string item;
        if (!(ls >> item) || item.rfind(name + "=", 0) != 0) fail("expected " + name + "=");
        try {
            return std::stod(item.substr(name.size() + 1));
        } catch (const std::exception&) {
            fail("bad value in '" + item + "'");
        }
        return 0.0;
    };
    while (std::getline(in, line)) {
        ++lineno;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string word;
        ls >> word;
        if (!header) {
            int version = 0;
            if (word != "nano-ep-trace" || !(ls >> version) || version != 1) fail("expected 'nano-ep-trace 1'");
            header = true;
        } else if (word == "key") {
            t.key = ProfileKey::parse(line.substr(4));
            have_key = true;
        } else if (word == "options") {
            t.options.overlap = field(ls, "overlap") != 0;
            t.options.lookahead = static_cast<int>(field(ls, "lookahead"));
            t.iters = static_cast<int>(field(ls, "iters"));
        } else if (word == "expert_group" || word == "order") {
            std::vector<int>& v = word == "order" ? t.groups.order : t.groups.expert_group;
            for (int g; ls >> g;) v.push_back(g);
        } else if (word == "rank") {
            std::size_t r = 0;
            if (!(ls >> r) || r != t.ranks.size()) fail("ranks must be listed in order");
            TraceRank tr;
            tr.tokens = static_cast<int>(field(ls, "tokens"));
            tr.route = field(ls, "route");
            tr.counts = field(ls, "counts");
            tr.pipeline = field(ls, "pipeline");
            tr.reduce = field(ls, "reduce");
            tr.total = field(ls, "total");
            t.ranks.push_back(tr);
        } else if (word == "counts") {
            std::size_t r = 0;
            if (!(ls >> r) || r >= t.ranks.size()) fail("counts for an unknown rank");
            for (std::int64_t n; ls >> n;) t.ranks[r].expert_counts.push_back(n);
        } else if (word == "op") {
            TraceOp op;
            std::string kind;
            if (!(ls >> op.rank >> op.op >> kind >> op.step >> op.issue_begin >> op.issue_end >> op.complete)) {
                fail("bad op line");
            }
            op.kind = parse_op_kind(kind);
            t.ops.push_back(op);
        } else {
            fail("unknown line '" + word + "'");
        }
    }
    if (!header || !have_key) fail("missing header or key");
    if (static_cast<int>(t.ranks.size()) != t.key.world) fail("rank count differs from key world");
    for (const TraceRank& r : t.ranks) {
        if (static_cast<int>(r.expert_counts.size()) != t.key.num_experts) fail("counts row size");
    }
    planner::validate_group_plan(t.groups, t.key.num_experts);
    return t;
}

LayerTrace LayerTrace::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open trace " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse(ss.str(), path);
}

void LayerTrace::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write trace " + path);
    f << serialize();
    if (!f) throw std::runtime_error("short write to " + path);
}

}  // namespace nano_nccl::exec
