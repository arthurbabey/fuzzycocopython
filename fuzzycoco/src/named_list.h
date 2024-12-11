#ifndef NAMED_LIST_H
#define NAMED_LIST_H

// implement a named list (like in R)

#include <iostream>
#include <string>
#include <memory>
#include <cassert>
#include <variant>
#include <vector>
using namespace std;


class Scalar : public variant<monostate, int, double, string> {
public:
    using variant<monostate, int, double, string>::variant;
    // inherit constructors

    bool is_int() const { return holds_alternative<int>(*this);  }
    bool is_double() const { return holds_alternative<double>(*this);  }
    bool is_string() const { return holds_alternative<string>(*this);  }
    bool is_null() const { return holds_alternative<monostate>(*this); }

    const string& get_string() const { return get<string>(*this); }
    int get_int() const { return get<int>(*this); }
    double get_double() const { return get<double>(*this); }

    bool operator==(int i) const { return is_int() &&  get<int>(*this) == i; }
    bool operator==(double d) const { return is_double() &&  get<double>(*this) == d; }
    bool operator==(const string& s) const { return is_string() &&  get<string>(*this) == s; }


    void print(ostream& out) const;
    static Scalar parse(istream& in);

    friend ostream& operator<<(ostream& out, const Scalar& scalar) {
        scalar.print(out);
        return out;
    }
};

class NamedList {
public:
    NamedList() {}
    NamedList(const string& name) { _name = name; }
    NamedList(const string& name, int i)  { _name = name; _value = i; }
    NamedList(const string& name, const string& s)  { _name = name; _value = s; }
    NamedList(const string& name, double v) { _name = name; _value = v; }
    NamedList(const string& name, const Scalar& scalar) : _name(name), _value(scalar) {}

    bool empty() const { return is_list() && size() == 0; }
    bool is_scalar() const { return !_value.is_null(); }
    bool is_list() const { return !is_scalar(); }
    int size() const { return _children.size(); }

    const NamedList& operator[](int idx) const { return *_children.at(idx); }
    NamedList& operator[](int idx)  { return *_children.at(idx); }
    // N.B: not optimized: check all names and return the first found...
    // if not found throw runtime_error
    const shared_ptr<NamedList> fetch(const string& name) const;
    const Scalar& fetch_scalar(const string& name) const { return fetch(name)->scalar(); }

    const NamedList& get_list(const string& name) const { return *fetch(name); }

    const string& get_string(const string& name) const { return fetch_scalar(name).get_string(); }
    int get_int(const string& name) const { return fetch_scalar(name).get_int(); }
    double get_double(const string& name) const { return fetch_scalar(name).get_double(); }

    // iterators
    // Define the iterator type (use underlying std::vector's iterator for simplicity)
    using iterator = vector<shared_ptr<NamedList>>::iterator;
    using const_iterator = vector<shared_ptr<NamedList>>::const_iterator;
    iterator begin() { return _children.begin(); }
    iterator end() { return _children.end(); }
    const_iterator begin() const { return _children.begin(); }
    const_iterator end() const { return _children.end(); }

    const Scalar& scalar() const { return _value; }
    const Scalar& value() const { return _value; }
    const string& name() const { return _name; }

    vector<string> names() const;

    void add(const string& name, int i) { add({name, i}); }
    void add(const string& name, double d) { add({name, d}); }
    void add(const string& name, const string& s) { add({name, s}); }

    void add(const string& name, const NamedList& elt);

    void print(ostream& out, int indent = 0, bool toplevel = true) const;

    bool operator==(const NamedList& l) const;

    static NamedList parse(istream& in);

    friend ostream& operator<<(ostream& out, const NamedList& list) {
        list.print(out);
        return out;
    }

protected:
    void add(const NamedList& elt) {
        assert(is_list());
        _children.push_back(make_shared<NamedList>(elt));
    }

private:
    string _name;
    Scalar _value;
    vector<shared_ptr<NamedList>> _children;
};

#endif // NAMED_LIST_H
