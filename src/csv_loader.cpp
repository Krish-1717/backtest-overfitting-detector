// csv_loader.cpp -- Implementation of CSV strategy return matrix loader.
// Day 2: Parses CSV files containing strategy return data for PSR/DSR analysis.

#include "csv_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace bopt {

static std::vector<std::string> split_line(const std::string& line, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, delim)) {
        // Trim whitespace and carriage returns
        while (!token.empty() && (token.back() == '\r' || token.back() == ' '))
            token.pop_back();
        while (!token.empty() && (token.front() == ' '))
            token.erase(token.begin());
        tokens.push_back(token);
    }
    return tokens;
}

CsvData load_csv(const std::string& filepath, bool has_header, char delimiter) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("csv_loader: cannot open file: " + filepath);
    }

    CsvData data;
    data.n_periods    = 0;
    data.n_strategies = 0;

    std::string line;
    bool first_row = true;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto tokens = split_line(line, delimiter);

        if (first_row && has_header) {
            data.headers = tokens;
            data.n_strategies = tokens.size();
            first_row = false;
            continue;
        }

        if (first_row) {
            data.n_strategies = tokens.size();
            first_row = false;
        }

        // Parse all tokens as doubles
        std::vector<double> row;
        row.reserve(tokens.size());
        for (const auto& tok : tokens) {
            try {
                row.push_back(std::stod(tok));
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "csv_loader: cannot parse value '" + tok +
                    "' as double in file: " + filepath
                );
            }
        }
        data.returns.push_back(row);
        ++data.n_periods;
    }

    if (data.n_strategies == 0 && !data.returns.empty()) {
        data.n_strategies = data.returns[0].size();
    }

    validate_csv(data);
    return data;
}

std::vector<std::vector<double>> to_column_major(const CsvData& data) {
    // Transpose: result[col][row]
    std::vector<std::vector<double>> cols(data.n_strategies,
                                          std::vector<double>(data.n_periods, 0.0));
    for (std::size_t r = 0; r < data.n_periods; ++r) {
        for (std::size_t c = 0; c < data.n_strategies; ++c) {
            if (c < data.returns[r].size()) {
                cols[c][r] = data.returns[r][c];
            }
        }
    }
    return cols;
}

void validate_csv(const CsvData& data) {
    if (data.returns.empty()) return;
    const std::size_t expected_cols = data.returns[0].size();
    for (std::size_t i = 1; i < data.returns.size(); ++i) {
        if (data.returns[i].size() != expected_cols) {
            throw std::runtime_error(
                "csv_loader: row " + std::to_string(i) +
                " has " + std::to_string(data.returns[i].size()) +
                " columns, expected " + std::to_string(expected_cols)
            );
        }
    }
    if (!data.headers.empty() && data.headers.size() != expected_cols) {
        throw std::runtime_error(
            "csv_loader: header has " + std::to_string(data.headers.size()) +
            " columns but data has " + std::to_string(expected_cols)
        );
    }
}

} // namespace bopt
