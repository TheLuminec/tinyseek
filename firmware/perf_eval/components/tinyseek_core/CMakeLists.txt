# Portable pre/post-processing shared by the firmware and the host test
# (host_test/ builds the same sources with g++, no ESP-IDF involved).
idf_component_register(SRCS "src/preprocess.cpp" "src/decode.cpp"
                       INCLUDE_DIRS "include")
