#pragma once

#include "graph_data.hpp"


namespace hotpot::search {

class Compatibility {
public:
    virtual ~Compatibility() = default;
    virtual bool accept_node(Position query, Position target) const = 0;
    virtual bool accept_edge(EdgeIndex query, EdgeIndex target) const = 0;
    virtual bool may_accept_node(Position query, Position target) const = 0;
    virtual bool may_accept_edge(EdgeIndex query, EdgeIndex target) const = 0;
};

}  // namespace hotpot::search
