#pragma once
// csv_loader.h -- Parse strategy return matrices from CSV files.
// Day 2: Enables real-world usage by loading a matrix of strategy returns
// from a CSV file instead of generating synthetic data.
//
// Expected CSV format:
//   - First row: optional header (strategy names)
//   - Each subsequent row: one time period
//   - Each column: one strategy's return for that period
//
// Example:
//   strat_A,strat_B,strat_C
//   0.012,-0.003,0.007
//   -0.005,0.011,-0.002

#ifndef CSV_LOADER_H
#define CSV_LOADER_H

#include <string>
#include <vector>
#include <stdexcept>

namespace bopt {

// Result of loading a CSV file.
struct CsvData {
    std::vector<std::string>         headers;      // column names (empty if no header row)
    std::vector<std::vector<double>> returns;      // [row][col] = return value
    std::size_t                      n_periods;    // number of rows (time steps)
    std::size_t                      n_strategies; // number of columns (strategies)
};

// Load a strategy return matrix from a CSV file.
// filepath   : path to the CSV file
// has_header : if true, treats the first row as column headers
// delimiter  : field separator (default comma)
// Throws std::runtime_error if the file cannot be opened or is malformed.
CsvData load_csv(
    const std::string& filepath,
    bool               has_header = true,
    char               delimiter  = ','
);

// Convert CsvData to column-major layout: result[strategy][period]
std::vector<std::vector<double>> to_column_major(const CsvData& data);

// Validate that all rows have the same number of columns.
// Throws std::runtime_error on inconsistency.
void validate_csv(const CsvData& data);

} // namespace bopt

#endif // CSV_LOADER_H
