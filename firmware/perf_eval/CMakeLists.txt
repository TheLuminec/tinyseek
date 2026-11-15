# On-device evaluation of tinyseek detectors: latency, memory and PC parity.
# ESP-IDF v5.5.6 project; see README.md for the pinned versions and build steps.
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tinyseek_perf_eval)
