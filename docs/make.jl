using Documenter, SlimOptim

makedocs(sitename="Slim Optimization toolbox",
         doctest=false, clean=true,
         authors="Mathias Louboutin",
         pages = Any[
             "Home" => "index.md",
             "Tutorials" => Any[
                 "tutorials/01_denoising.md",
                 "tutorials/02_simple_constrained.md",
                 "tutorials/03_constr_fwi_judi.md"
             ]
         ])

deploydocs(repo="github.com/slimgroup/SlimOptim.jl")