cmake_minimum_required(VERSION 3.15)

# Assume the person building this is in industry unless the
# ACADEMIC=ON environmental variable is set

if(DEFINED ENV{ACADEMIC})
    if($ENV{ACADEMIC})
      set(ACADEMIC TRUE)
    else()
      set(ACADEMIC FALSE)
    endif()
  else()
    set(ACADEMIC FALSE)
endif()


if(NOT DEFINED FFPOPTVERSION)
  set(FFPOPTVERSION 0.0)
endif()


if(NOT DEFINED USE_FENNOL)
  set(USE_FENNOL TRUE)
endif()


if(NOT DEFINED USE_MACE)
  set(USE_MACE TRUE)
endif()


if(NOT DEFINED USE_PM6ML)
  set(USE_PM6ML TRUE)
endif()


if(NOT DEFINED USE_ESPALOMA)
  set(USE_ESPALOMA TRUE)
endif()


if(NOT DEFINED USE_HILFIKER)
  set(USE_HILFIKER TRUE)
endif()



if(NOT DEFINED USE_DFTB)
  if(DEFINED ENV{AMBERHOME})
    set(USE_DFTB TRUE)
    message(WARNING "USE_DFTB is TRUE because ENV{AMBERHOME} is defined $ENV{AMBERHOME}")
  endif()
endif()
if(USE_DFTB)
  if(NOT DEFINED ENV{AMBERHOME})
    set(USE_DFTB FALSE)
    message(WARNING "Disabling DFTB2/DFTB3 parameter downloads because AMBERHOME is unset")
  endif()
endif()


# Industry users cannot use certain models
# without consulting the devlopers

if(NOT ACADEMIC)
  set(USE_FENNOL FALSE)
  set(USE_MACE FALSE)
  set(USE_PM6ML FALSE)
endif()


if(NOT USE_MACE)
  set(USE_HILFIKER FALSE)
endif()


project(ffpopt VERSION "${FFPOPTVERSION}" LANGUAGES Fortran)
include(FetchContent)
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)
set(PackageName "ffpopt") # Define your package name
set(PKGDATA_DIR "${Python_SITELIB}/${PackageName}/pkgdata")



if(USE_FENNOL)


  if(FALSE)
  
    FetchContent_Declare(
      fennol_pmc
      GIT_REPOSITORY https://github.com/FeNNol-tools/FeNNol-PMC.git
      GIT_TAG        main # Or a specific commit hash/tag e.g., "v1.0.0"
      ENVIRONMENT GIT_LFS_SKIP_SMUDGE=1
    )
  
    # 2. Populate the content (downloads it)
    FetchContent_MakeAvailable(fennol_pmc)

    # At this point, the source directory is available in ${fennol_pmc_SOURCE_DIR}
    message("SOURCE_DIR = ${fennol_pmc_SOURCE_DIR}")
    file(GLOB SRC_FILES "${fennol_pmc_SOURCE_DIR}/*")
    message(STATUS "Files in the directory: ${SRC_FILES}")

    # Define source and destination paths
    set(FENNOL_MODEL_DIR "${fennol_pmc_SOURCE_DIR}/FENNIX-BIO1/v1.0")
    set(MODEL_M_PATH "${FENNOL_MODEL_DIR}/fennix-bio1M.fnx")
    set(MODEL_S_PATH "${FENNOL_MODEL_DIR}/fennix-bio1S.fnx")

  else()

    include(ExternalProject)

    # 1. Add the external projects
    ExternalProject_Add(fennol_model_m
      URL "https://github.com/FeNNol-tools/FeNNol-PMC/blob/main/FENNIX-BIO1/v1.0/fennix-bio1M.fnx"
      DOWNLOAD_NO_EXTRACT 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      PREFIX "${CMAKE_BINARY_DIR}/fennixmodels_m"
    )
    
    ExternalProject_Add(fennol_model_s
      URL "https://github.com/FeNNol-tools/FeNNol-PMC/blob/main/FENNIX-BIO1/v1.0/fennix-bio1S.fnx"
      DOWNLOAD_NO_EXTRACT 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      PREFIX "${CMAKE_BINARY_DIR}/fennixmodels_s"
    )
    
    # 2. Retrieve the actual download directory properties
    ExternalProject_Get_Property(fennol_model_m DOWNLOAD_DIR)
    set(MODEL_M_PATH "${DOWNLOAD_DIR}/fennix-bio1M.fnx")
    
    ExternalProject_Get_Property(fennol_model_s DOWNLOAD_DIR)
    set(MODEL_S_PATH "${DOWNLOAD_DIR}/fennix-bio1S.fnx")
    
  endif()
  
endif()


if(USE_ESPALOMA)

  include(ExternalProject)

  # 1. Add the external projects
  ExternalProject_Add(espaloma_model
    URL "https://github.com/choderalab/espaloma_charge/releases/download/v0.0.8/model.pt"
    DOWNLOAD_NO_EXTRACT 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    PREFIX "${CMAKE_BINARY_DIR}/espaloma_model"
  )
    
  # 2. Retrieve the actual download directory properties
  ExternalProject_Get_Property(espaloma_model DOWNLOAD_DIR)
  set(ESPALOMA_PATH "${DOWNLOAD_DIR}/model.pt")
    
  
endif()



if(USE_HILFIKER)

  include(ExternalProject)

  # 1. Add the external projects
  ExternalProject_Add(hilfiker_model
    URL "https://github.com/mathilfiker/ml_for_charges/raw/refs/heads/main/charge_prediction/model/xgb_chgs.json"
    DOWNLOAD_NO_EXTRACT 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    PREFIX "${CMAKE_BINARY_DIR}/hilfiker_model"
  )
    
  # 2. Retrieve the actual download directory properties
  ExternalProject_Get_Property(hilfiker_model DOWNLOAD_DIR)
  set(HILFIKER_PATH "${DOWNLOAD_DIR}/xgb_chgs.json")  
  
endif()


if(USE_MACE)
  
  FetchContent_Declare(
    mace_off
    GIT_REPOSITORY https://github.com/ACEsuit/mace-off.git
    GIT_TAG        main # Or a specific commit hash/tag e.g., "v1.0.0"
  )
  
  # 2. Populate the content (downloads it)
  FetchContent_MakeAvailable(mace_off)

  # At this point, the source directory is available in ${mace_off_SOURCE_DIR}
  message("SOURCE_DIR = ${mace_off_SOURCE_DIR}")
  file(GLOB SRC_FILES "${mace_off_SOURCE_DIR}/*")
  message(STATUS "Files in the directory: ${SRC_FILES}")

  # Define source and destination paths
  set(MACE_MODEL_DIR "${mace_off_SOURCE_DIR}")
  
endif()



if(USE_PM6ML)
  
  FetchContent_Declare(
    pm6ml
    GIT_REPOSITORY https://github.com/Honza-R/mopac-ml.git
    GIT_TAG        main # Or a specific commit hash/tag e.g., "v1.0.0"
  )
  
  # 2. Populate the content (downloads it)
  FetchContent_MakeAvailable(pm6ml)

  # At this point, the source directory is available in ${mace_off_SOURCE_DIR}
  message("SOURCE_DIR = ${pm6ml_SOURCE_DIR}")
  file(GLOB SRC_FILES "${pm6ml_SOURCE_DIR}/*")
  message(STATUS "Files in the directory: ${SRC_FILES}")

  # Define source and destination paths
  set(PM6ML_MODEL_DIR "${pm6ml_SOURCE_DIR}/models")
  
endif()


#############################
# DFTB2/DFTB3 parameters
#############################
if(USE_DFTB)
  
  FetchContent_Declare(
    dftb3obparams
    GIT_REPOSITORY https://github.com/dftbparams/3ob
    GIT_TAG        main # Or a specific commit hash/tag e.g., "v1.0.0"
  )
  
  # 2. Populate the content (downloads it)
  FetchContent_MakeAvailable(dftb3obparams)

  # At this point, the source directory is available in ${dftb3obparams_SOURCE_DIR}
  message("SOURCE_DIR = ${dftb3obparams_SOURCE_DIR}")
  file(GLOB SRC_FILES "${dftb3obparams_SOURCE_DIR}/skfiles/*.skf")
  message(STATUS "Files in the directory: ${SRC_FILES}")

  # Define source and destination paths
  set(DFTB3OB_FILES "${SRC_FILES}")


  
  FetchContent_Declare(
    dftbmioparams
    GIT_REPOSITORY https://github.com/dftbparams/mio
    GIT_TAG        main # Or a specific commit hash/tag e.g., "v1.0.0"
  )
  
  # 2. Populate the content (downloads it)
  FetchContent_MakeAvailable(dftbmioparams)

  # At this point, the source directory is available in ${dftbmioparams_SOURCE_DIR}
  message("SOURCE_DIR = ${dftbmioparams_SOURCE_DIR}")
  file(GLOB SRC_FILES "${dftbmioparams_SOURCE_DIR}/skfiles/*.skf")
  message(STATUS "Files in the directory: ${SRC_FILES}")

  # Define source and destination paths
  set(DFTBMIO_FILES "${SRC_FILES}")
  
endif()


#####################################################################
# BELOW IS WHEN WE'RE BUILDING VIA SKBUILD/pip
#####################################################################
if(DEFINED SKBUILD_SCRIPTS_DIR AND NOT SKBUILD_SCRIPTS_DIR STREQUAL "")

  
  # 1. Compile the Fortran source
  # This creates the binary in the build folder
  add_executable(ffpopt-respf src/resp/resp.f)
  
  set_target_properties(ffpopt-respf PROPERTIES
    Fortran_STANDARD 90
    Fortran_STANDARD_REQUIRED ON
    Fortran_EXTENSIONS OFF
  )
  
  # 2. Tell CMake to install the binary to the Python bin folder
  # SKBUILD_SCRIPTS_DIR is provided by scikit-build-core
  install(TARGETS ffpopt-respf DESTINATION "${SKBUILD_SCRIPTS_DIR}")
  
  # # 3. Handle the pre-compiled binary
  # install(FILES "src/mopac-ml/ffpopt-mopacml"
  #   DESTINATION "${SKBUILD_SCRIPTS_DIR}"
  #   PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
  #   GROUP_READ GROUP_EXECUTE 
  #   WORLD_READ WORLD_EXECUTE)

  
  if(USE_FENNOL)
    install(FILES "${MODEL_M_PATH}" "${MODEL_S_PATH}"
      DESTINATION "ffpopt/pkgdata/fennix"
    )
  endif()


  if(USE_ESPALOMA)
    install(FILES "${ESPALOMA_PATH}"
      DESTINATION "ffpopt/pkgdata/espaloma"
    )
  endif()

  
  if(USE_HILFIKER)
    install(FILES "${HILFIKER_PATH}"
      DESTINATION "ffpopt/pkgdata/ml_for_charges"
    )
  endif()

  
  if(USE_MACE)
    install(FILES "${MACE_MODEL_DIR}/mace_off23/MACE-OFF23_small.model"
      "${MACE_MODEL_DIR}/mace_off23/MACE-OFF23_medium.model"
      "${MACE_MODEL_DIR}/mace_off23/MACE-OFF23_large.model"
      DESTINATION "ffpopt/pkgdata/mace-off/mace_off23"
    )
    install(FILES "${MACE_MODEL_DIR}/mace_off24/MACE-OFF24_medium.model"
      DESTINATION "ffpopt/pkgdata/mace-off/mace_off24"
    )
  endif()


  if(USE_PM6ML)
    install(FILES "${PM6ML_MODEL_DIR}/PM6-ML_correction_seed8_best.ckpt"
      DESTINATION "ffpopt/pkgdata/pm6ml"
    )
  endif()

  
  if(USE_DFTB)
    file(COPY ${DFTBMIO_FILES} DESTINATION "$ENV{AMBERHOME}/dat/slko/mio-1-1")
    file(COPY ${DFTB3OB_FILES} DESTINATION "$ENV{AMBERHOME}/dat/slko/3ob-3-1")
  endif()

  
  #####################################################################
else() # BELOW IS WHEN WE'RE NOT BUILDING VIA SKBUILD
  #####################################################################

  
  add_subdirectory(src/resp)
  #add_subdirectory(src/mopac-ml)
  if(BUILD_PYTHON)
    add_subdirectory(src/python)
  endif()

  if(USE_FENNOL)
    file(COPY
      "${MODEL_M_PATH}"
      "${MODEL_S_PATH}"
      DESTINATION "${PKGDATA_DIR}/fennix")
  endif()

  if(USE_ESPALOMA)
    file(COPY
      "${ESPALOMA_PATH}"
      DESTINATION "${PKGDATA_DIR}/espaloma")
  endif()

  if(USE_HILFIKER)
    file(COPY
      "${HILFIKER_PATH}"
      DESTINATION "${PKGDATA_DIR}/ml_for_charges")
  endif()

  if(USE_MACE)
    file(COPY "${MACE_MODEL_DIR}/mace_off23/MACE-OFF23_small.model"
      "${MACE_MODEL_DIR}/mace_off23/MACE-OFF23_medium.model"
      "${MACE_MODEL_DIR}/mace_off23/MACE-OFF23_large.model"
      DESTINATION "${PKGDATA_DIR}/mace-off/mace_off23")
    file(COPY "${MACE_MODEL_DIR}/mace_off24/MACE-OFF24_medium.model"
      DESTINATION "${PKGDATA_DIR}/mace-off/mace_off24")
  endif()
  
  if(USE_PM6ML)
    file(COPY
      "${PM6ML_MODEL_DIR}/PM6-ML_correction_seed8_best.ckpt"
      DESTINATION "${PKGDATA_DIR}/pm6ml")
  endif()


  if(USE_DFTB)
    file(COPY ${DFTBMIO_FILES} DESTINATION "$ENV{AMBERHOME}/dat/slko/mio-1-1")
    file(COPY ${DFTB3OB_FILES} DESTINATION "$ENV{AMBERHOME}/dat/slko/3ob-3-1")
  endif()

endif()
