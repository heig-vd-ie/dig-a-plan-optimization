# How to create julia projects

Create a Julia Project (with its own environment)
From your project folder (in WSL or anywhere):

```sh
mkdir MyProject
cd MyProject
julia --project=.
```
Then inside the Julia REPL:

```julia
import Pkg
Pkg.activate(".")     # Activate this folder as a project
Pkg.add("JuMP")
Pkg.add("SDDP")
Pkg.add("Plots")
```
This will create two files in your folder:

Project.toml – lists dependencies

Manifest.toml – pins exact versions (optional to commit)

2. Always Run Code Using the Project Environment
In your Julia script (main.jl), start with:

```julia
# main.jl
using Pkg
Pkg.activate(@__DIR__)     # Activates the environment in this folder
Pkg.instantiate()          # Installs all packages if not already installed

using JuMP, SDDP, Plots

println("Everything is ready!")
```

🧠 This ensures your script can be run from anywhere and still use the local project environment.

3. Run Your Script (Anywhere, Any Time)
From WSL or terminal:

```sh
julia main.jl
```

Or, open VS Code in the folder and just hit Run – it'll use the local environment.

4. How to run current code: 

`julia --project=. any_file.jl`