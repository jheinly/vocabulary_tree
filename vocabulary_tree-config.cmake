cmake_minimum_required(VERSION 3.0)

# This code depends on cmake_helper (https://github.com/jheinly/cmake_helper).
find_package(cmake_helper REQUIRED)

# Uncomment the following line if this module should be treated as a 3rd-party
# module (code that is not editable). 3rd-party modules have a different
# compiler warning level set by default.
#cmh_set_as_third_party_module()

cmh_new_module_with_dependencies()
