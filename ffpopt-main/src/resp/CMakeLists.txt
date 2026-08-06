file(GLOB SOURCES *.f)
add_executable(resp ${SOURCES})
set_target_properties(resp PROPERTIES 
  OUTPUT_NAME "ffpopt-respf"
  INSTALL_RPATH \$ORIGIN/../lib:\$ORIGIN/../lib64
  )
target_compile_definitions(resp PRIVATE PACKAGE_VERSION="${CMAKE_PROJECT_VERSION}")
install(TARGETS resp DESTINATION ${CMAKE_INSTALL_BINDIR})
