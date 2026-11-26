# Portable pre/post-processing shared by the firmware and the host test
# (host_test/ builds the same sources with g++, no ESP-IDF involved). The Arduino sketch
# (firmware/perf_eval_arduino) gets generated copies: python -m tinyseek.deploy.arduino_sketch
idf_component_register(SRCS "src/preprocess.cpp" "src/decode.cpp" "src/jpeg_info.cpp" "src/kv.cpp"
                       INCLUDE_DIRS "include")
