#include <iomanip>

#include "named_list.h"
#include "types.h"
#include "string_utils.h"

void Scalar::print(ostream& out) const {
    if  (holds_alternative<monostate>(*this))
        out << "-";
    else if (holds_alternative<int>(*this))
        out << get_int();
    else if (holds_alternative<double>(*this))
        out << StringUtils::prettyDistinguishableDoubleToString(get_double());
    else out << quoted(get_string());
}

Scalar Scalar::parse(istream& in) {
    const char QUOTE = '"';
    string item;
    in >> item;
    if (item.at(0) == QUOTE) { // string
        int last_quote_idx = item.find_last_of(QUOTE);
        string s(item.begin() + 1, item.begin() + last_quote_idx);
        return Scalar(s);
    }

    // look for the dot
    if (item.find_first_of('.') != item.npos) // double
        return Scalar(stod(item));

    return Scalar(stoi(item));
}


bool NamedList::operator==(const NamedList& l) const {
    if (_name != l._name || _value != l._value) return false;

    const int nb = _children.size();
    if (nb != l._children.size()) return false;
    for (int i = 0; i < nb; i++) {
        if (! (*_children[i] == *l._children[i]) ) return false;
    }


    return true;
}

NamedList NamedList::parse(istream &in)
{
    const char QUOTE = '"';
    const char SPACE = ' ';
    const char COMMA = ',';

    auto peek_next_significant = [&](){ while(in.peek() <= SPACE) in.get(); return in.peek(); };
    auto skip_comma = [&](){ if (in.peek() == COMMA) in.get(); };

    char ch = peek_next_significant();
    if (ch == '}')
    { // end of list
        in.get(); // consume }
        return NamedList();
    }

    if (ch == '{')
    { // list
        NamedList list;
        in.get(); // consume {
        while (peek_next_significant() != '}')
        {
            auto child = NamedList::parse(in);
            skip_comma();
            list.add(child);
        }
        in.get(); // consume }
        return list;
    }

    if (ch == QUOTE)
    { // scalar
        char c = 0;
        in.get(c); // consume "
        string elt_name;
        getline(in, elt_name, QUOTE);
        in.get(c); // consume =
        assert(c == ':');
        if (peek_next_significant() == '{') {
            auto sublist = NamedList::parse(in);
            skip_comma();
            sublist._name = elt_name;
            return sublist;
        }
        auto scalar = Scalar::parse(in);
        skip_comma();
        return NamedList(elt_name, scalar);
    }

    return NamedList();
}

const shared_ptr<NamedList> NamedList::fetch(const string& name) const {
    int nb = size();

    for (const auto& child : _children)
        if (child->_name == name) return child;

    THROW_WITH_LOCATION("name not found in list: " + name);
    return _children[0]; // to avoid a compiler warning
}

vector<string> NamedList::names() const {
    vector<string> res;
    res.reserve(size());
    for (const auto& child : _children)
        res.push_back(child->_name);
    return res;
}

void NamedList::print(ostream& out, int indent, bool toplevel) const {
    char SPACE = ' ';
    int INDENT = 2;
    string SPACER(indent, SPACE);
    out << SPACER;
    if (!name().empty())
        out << quoted(name()) << ":";
    if (is_list()) {
        out << "{";
        if (!empty())
            out << endl;
        const int nb = size();
        for (int i = 0; i < nb; i++) {
            _children[i]->print(out, indent + INDENT, false);
            if (i != (nb - 1))
                out << ",";
            out << endl;
        }
        if (!empty())
            out << SPACER;
        out << "}";
    } else {
        _value.print(out);
        // out << endl;
    }
    if (toplevel)
        out << endl;

}

void NamedList::add(const string& name, const NamedList& elt) {
    assert(is_list());
    auto node = make_shared<NamedList>(elt);
    node->_name = name;
    _children.push_back(node);
}
