/**
  * @file   fuzzyset.h
  * @author Jean-Philippe Meylan <jean-philippe.meylan_at_heig-vd.ch>
  * @author ReDS (Reconfigurable and embedded digital systems) <www.reds.ch>
  * @author HEIG-VD (Haute école d'ingénierie et de gestion) <www.heig-vd.ch>
  * @date   07.2009
  * @section LICENSE
  *
  * This application is free software; you can redistribute it and/or
  * modify it under the terms of the GNU Lesser General Public
  * License as published by the Free Software Foundation; either
  * version 2.1 of the License, or (at your option) any later version.
  *
  * This library is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  * Lesser General Public License for more details.
  *
  * You should have received a copy of the GNU Lesser General Public
  * License along with this library; if not, write to the Free Software
  * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
  *
  * @class FuzzySet
  *
  * @brief This class represent a simple fuzzy coco set. It is defined by a name and a parameter
  * which indicates the position of the set. See FuzzyCocoMemberships for more detail.
  */

#ifndef FUZZYSET_H
#define FUZZYSET_H

#include <string>
#include <iomanip>
using namespace std;
#include "types.h"
#include "named_list.h"
#include "string_utils.h"

class FuzzySet
{
public:
    FuzzySet(string name, double pos = MISSING_DATA_DOUBLE) : _name(name), _position(pos) {}
    FuzzySet(const FuzzySet& set) : _name(set._name), _position(set._position) {}
    FuzzySet(const FuzzySet&& set) : _name(move(set._name)), _position(set._position) {}
    ~FuzzySet() {}
    FuzzySet& operator=(FuzzySet set) { 
      swap(_name, set._name);
      _position = set._position;
      return *this;
    }

    const string& getName() const { return _name; }
    // Retrieve the position of the set.
    double getPosition() const {return _position; }
    // Set the position of the set.
    void setPosition(double pos) { _position = pos; }
    void setName(const string& name) { _name = name; }

    NamedList describe() const {
      NamedList desc;
      desc.add("name", getName());
      desc.add("position", getPosition());
      string pretty = getName() + "@";
      if (is_na(getPosition()))
        pretty += "NA";
      else
        pretty += StringUtils::prettyDistinguishableDoubleToString(getPosition());
      desc.add("pretty", pretty);
      return desc;
    }

    static void printDescription(ostream& out, const NamedList& desc) {
      out << quoted(desc.get_string("pretty"));
    }

    // for debug purposes
    bool operator==(const FuzzySet& set) const { 
      return getName() == set.getName() && getPosition() == set.getPosition();
    } 

private:
    string _name;
    double _position;
};

inline ostream& operator<<(ostream& out, const FuzzySet& set) {
  set.printDescription(out, set.describe());
  return out;
}

#endif // FUZZYSET_H
