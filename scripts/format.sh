#!/bin/bash
find . -name "*.hpp" -o -name "*.cpp" | grep -v build | xargs clang-format-14 -i