idf_component_register(
    SRCS "perf_eval.cpp" "platform.cpp" "camera.cpp"
    # board_pins.h is shared with the capture sketch so both firmwares use the same wiring.
    INCLUDE_DIRS "." "../../capture"
    # esp-dl, esp32-camera and esp_new_jpeg come from idf_component.yml, which makes them
    # requirements of this component automatically.
    REQUIRES tinyseek_core fatfs sdmmc esp_driver_sdmmc esp_timer esp_hw_support)

# Record the component versions actually resolved by the component manager in every CSV row.
function(tinyseek_component_version component out)
    set(manifest "${CMAKE_SOURCE_DIR}/managed_components/${component}/idf_component.yml")
    set(version "unknown")
    if(EXISTS "${manifest}")
        file(STRINGS "${manifest}" lines REGEX "^version:")
        if(lines)
            list(GET lines 0 line)
            string(REGEX REPLACE "^version: *\"?([^\"]*)\"?.*$" "\\1" version "${line}")
        endif()
    endif()
    set(${out} "${version}" PARENT_SCOPE)
endfunction()

tinyseek_component_version(espressif__esp-dl ESPDL_VERSION)
tinyseek_component_version(espressif__esp32-camera CAMERA_VERSION)
tinyseek_component_version(espressif__esp_new_jpeg JPEG_VERSION)
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    TINYSEEK_ESPDL_VERSION="${ESPDL_VERSION}"
    TINYSEEK_CAMERA_VERSION="${CAMERA_VERSION}"
    TINYSEEK_JPEG_VERSION="${JPEG_VERSION}")
