#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <sstream>
#include <string>
#include <algorithm>
using namespace std;
namespace StringUtils {
    // remove trailing zeroes except the one just after the dot
    // i.e 1.100 --> 1.1 but 1.00000 -> 1.0
    // --> a double is distinguishable from an integer
    inline string prettyDistinguishableDoubleToString(double value) {
        ostringstream oss;
        // Use fixed and setprecision to avoid scientific notation
        oss << fixed << value;

        string result = oss.str();

        int idx = result.find_last_not_of('0') + 1;
        int dot_idx = result.find_last_of('.');
        int zeroes_idx = max(idx, dot_idx + 2);
        if (zeroes_idx < result.length()) {
            result.erase(zeroes_idx, string::npos);
        }
        return result;
    }
}

#endif // STRING_UTILS_H
