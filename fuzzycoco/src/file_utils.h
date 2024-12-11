/**
  * @file   file_utils.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @date   09.2024
  * @brief misc utilities
  */


#ifndef FUZZY_COCO_UTILS_H
#define FUZZY_COCO_UTILS_H

#include <string>
#include <vector>
#include <filesystem>
using namespace std;
using namespace std::filesystem;
#include "dataframe.h"

namespace FileUtils
{

  void mkdir_if_needed(path file_path);

  void parseCSV(const string& content, vector<vector<string>>& tokens, char delim = ';');
  void parseCSV(istream& in, vector<vector<string>>& tokens, char delim = ';');
  void parseCSV(const path& filename, vector<vector<string>>& tokens, char delim = ';');

  void writeCSV(ostream& out, const DataFrame& df, char delim = ';');

  string slurp(const path& filename);

};




#endif // FUZZY_COCO_UTILS_H
