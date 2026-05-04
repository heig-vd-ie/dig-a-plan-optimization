# Ray Worker Setup (Remote Execution)

This guide explains how to connect a **remote machine** as a **Ray worker** to run heavy tasks for Dig-A-Plan.

The typical setup uses **Tailscale** to create a private network between machines.

---

## 1) Concepts

- **Head machine**: runs the main stack (Python/FastAPI, Julia SDDP service, Ray head, dashboards).
- **Worker machine**: joins the Ray cluster and executes Ray tasks.
- **Tailscale IP**: the “private address” that Tailscale gives to each machine (it often looks like `100.xx.yy.zz`).  
Use this address to connect machines together (SSH to the worker, and connect the Ray worker to the Ray head), even if they are on different networks.

---

## 2) Install and setup Tailscale

Tailscale creates a private network between the **head** and **worker** machines, so they can communicate even if they are on different networks.

### Linux / Ubuntu / WSL

1. **Install Tailscale**
```sh
   curl -fsSL https://tailscale.com/install.sh | sh
```
2. **Start Tailscale and log in**
```sh
   sudo tailscale up
```
Follow the login link shown in the terminal. Make sure both machines are connected to the same Tailscale account/team.

3. **Get the Tailscale IP**
```sh
   tailscale ip
```
You will get an IP like 100.xx.yy.zz. When starting the Ray worker, use the head machine’s Tailscale IP as HEAD_HOST

---

## 3) Prerequisites

### On both machines
- The project is set up on both machines (e.g. run `make install-all` and then you can run `make venv-activate` and the required tools are installed).
- Both machines can reach each other over a private network, it means that both are connected to the same **Tailscale** network (same account/team). You can verify connectivity with:
``` sh
    tailscale status
```

### On the worker machine
- You can SSH into the worker from the head machine. For this, you need an address like `user@worker-host` or `user@<worker-ip>`.

### Required environment variable
- `SERVER_RAY_PORT` must be set (usually via **.envrc**). If you use **direnv**, run:

``` sh
    direnv allow
```

---


## 4) Start the stack on the head machine

On the **head** machine:
```sh
make start
```

This typically:
- starts Docker services (Python/FastAPI, Julia SDDP service, Grafana/Prometheus/Mongo),
- opens a `tmux` session with panes for logs and an interactive shell.

---

## 5) Start the Ray worker (on the worker machine)

Once you are **logged in to the worker machine** (SSH session), start the worker:

```sh
make run-ray-worker
```
This command runs the script ./scripts/start-ray-worker.sh.

That script will ask you:
```sh
    “Enter head host IP (e.g., 100.66.5.3):”
```
You need to type the IP address of the head machine that the worker can reach.

The script will runs:

```sh
    ray start --address="HEAD_IP:SERVER_RAY_PORT"
```
which means that start Ray on this worker and connect it to the Ray head running at HEAD_IP on port SERVER_RAY_PORT. 

Then it continuously shows cluster status every 5 seconds:

```sh
    watch -n 5 ray status
```

---

## 6) Verify the worker is connected

### On the worker machine
After you run `make run-ray-worker`, the script shows `ray status` every few seconds.  
You should see the worker listed as an active node in the cluster.

If you don’t see it, the worker is not connected to the head yet (check the Troubleshooting section).

### On the head machine
You can confirm the cluster from the Ray dashboard:
- http://localhost:8265

If your Ray dashboard uses a different port in your setup, use that configured port.

---



## 7) Troubleshooting

### “Need to set SERVER_RAY_PORT”
Your environment is missing `SERVER_RAY_PORT`.
- Ensure `.envrc` contains it
- Run:
```sh
  direnv allow
```
- Or export it manually in your shell.

```sh
    export SERVER_RAY_PORT
```

### Worker cannot connect to head
- Confirm both machines are on the **same Tailscale network**:
```sh
  tailscale status
```
- Ensure you used the correct **head IP** .
- Verify the Ray port is reachable (firewall/ufw). If needed on the head:
```sh
  make permit-remote-ray-port
```

### Ray is stuck / old sessions exist
On the machine with the issues:

```sh
    make clean-ray
```


## 8) Notes for maintainers

- `make connect-ray-worker` runs `./scripts/connect-ray-worker.sh` (prompts for `user@worker` then runs `ssh`).
- `make run-ray-worker` runs `./scripts/start-ray-worker.sh` (interactive head IP prompt and `ray start --address=...`).
- `make start` runs `./scripts/start-servers.sh` and opens a tmux session (logs + worker + interactive panes).
