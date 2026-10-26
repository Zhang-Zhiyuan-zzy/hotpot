#pragma once

#include "../../graph/_native/graph.hpp"


namespace hotpot::search {

using hotpot::graph::EdgeIndex;
using hotpot::graph::Position;

class Compatibility {
public:
    virtual ~Compatibility() = default;
    virtual bool accept_node(Position query, Position target) const = 0;
    virtual bool accept_edge(EdgeIndex query, EdgeIndex target) const = 0;
    virtual bool may_accept_node(Position query, Position target) const = 0;
    virtual bool may_accept_edge(EdgeIndex query, EdgeIndex target) const = 0;
};

}  // namespace hotpot::search
