# Install Julia

## In WSL
### Step-by-Step Setup: Julia in VS Code on WSL
#### Prerequisites

WSL installed (preferably WSL2)

Ubuntu (or other Linux) running on WSL

VS Code with the Remote - WSL extension

#### ✅ Step 1: Install Julia inside WSL
Open your WSL terminal (e.g., Ubuntu):

```sh
mkdir Downloads
cd /Downloads
sudo apt install curl
curl -L -O https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.0-linux-x86_64.tar.gz
tar -xvzf julia-1.11.0-linux-x86_64.tar.gz
sudo mv julia-1.11.0 /opt/
sudo ln -s /opt/julia-1.11.0/bin/julia /usr/local/bin/julia
```

Check that Julia is installed:

```sh
julia --version
```


#### ✅ Step 2: Install Julia Extension in VS Code (WSL)
Inside VS Code (running in WSL), go to Extensions (Ctrl+Shift+X)

Install the "Julia" extension (by Julia Computing)

It should auto-detect Julia installed in /usr/local/bin/julia

If it doesn't, set it manually in settings.json (inside WSL):

```json
"julia.executablePath": "/usr/local/bin/julia"
```

To open settings.json, use Ctrl+Shift+P → "Preferences: Open Settings (JSON)"

#### ✅ Step 3: Try Running Julia Code in WSL
In VS Code, create a file test.jl

```julia
println("Hello from Julia in WSL!")
```

Press Ctrl+Shift+Enter or click Run → Output should show in the Julia REPL terminal


#### ✅ Step 4: Install Packages (from WSL terminal or REPL)

```julia
using Pkg
Pkg.add("JuMP")
Pkg.add("SDDP")
Pkg.add("Plots")
```

Packages will install in your WSL file system, isolated from Windows Julia (if you also have it there).

