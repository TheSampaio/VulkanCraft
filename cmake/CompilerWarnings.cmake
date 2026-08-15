function(vulkancraft_set_warnings target_name)
    target_compile_options(${target_name} PRIVATE /W4 /permissive- /EHsc)
endfunction()
