using DifferentialEquations  # https://diffeq.sciml.ai/stable/
using PyPlot  # https://github.com/JuliaPy/PyPlot.jl
using StatsBase  # https://juliastats.org/StatsBase.jl/stable/
using ProgressMeter  # https://github.com/timholy/ProgressMeter.jl

# Set font of the plots to Arial
rc("font", family="arial")

# Helper function for slicing PyPlot figures
pyslice(i1, i2) = PyPlot.pycall(PyPlot.pybuiltin("slice"), PyPlot.PyObject, i1, i2)

## ----------------------------------------------------------------------------
## Create the model -----------------------------------------------------------
## ----------------------------------------------------------------------------

# Model equations
function model!(du, u, p, t)

    # State variables
    v, w, x, y, z, z2, v2, w2, x2, y2 = u

    # Parameters
    τ, ε, α, β, γ, k, v₀, w₀, x₀, k1, k2, kz, kcext, Cmedia, D = p

    # Equations
    du[1] = dv = τ * ( v₀ - α*v + β*v^2 - v^3 - w*v + v*z )
    du[2] = dw = τ * ( ε*(w₀ + (γ*v^2 / (v^2 + k)) - w) )
    du[3] = dx = τ * ( x₀ - k1 * v^2 * x )
    du[4] = dy = τ * ( k1 * v^2 * x - k2 * (1/(v^3 + 1)) * y )
    du[5] = dz = τ * ( -kcext * (z - Cmedia) + k2 * (1/(v^3 + 1)) * y - v*z + kz*exp(-D)*z2 - kz*exp(-D)*z )
    du[6] = dz2 = τ * ( -kcext * (z2 - Cmedia) + k2 * (1/(v2^3 + 1)) * y2 - v2*z2 + kz*exp(-D)*z - kz*exp(-D)*z2 )
    du[7] = dv2 = τ * ( v₀ - α*v2 + β*v2^2 - v2^3 - w2*v2 + v2*z2 )
    du[8] = dw2 = τ * ( ε*(w₀ + (γ*v2^2 / (v2^2 + k)) - w2) )
    du[9] = dx2 = τ * ( x₀ - k1 * v2^2 * x2 )
    du[10] = dy2 = τ * ( k1 * v2^2 * x2 - k2 * (1/(v2^3 + 1)) * y2 )

end

# Noise function
function noise!(du, u, p, t)

    # State variables
    v, w, x, y, z, z2, v2, w2, x2, y2 = max.(u, 0)

    # Parameters
    τ, ε, α, β, γ, k, v₀, w₀, x₀, k1, k2, kz, kcext, Cmedia, D, Ω = p

    # Equations
    σ = 1/sqrt(Ω)

    du[1, 1] = σ * sqrt(τ * v₀)
    du[1, 2] = σ * sqrt(τ * α*v)
    du[1, 3] = σ * sqrt(τ * β*v^2)
    du[1, 4] = σ * sqrt(τ * v^3)
    du[1, 5] = σ * sqrt(τ * w*v)
    du[1, 6] = σ * sqrt(τ * v*z)

    du[2, 7] = σ * sqrt(τ * ε*w₀)
    du[2, 8] = σ * sqrt(τ * ε*γ*v^2 / (v^2 + k))
    du[2, 9] = σ * sqrt(τ * ε*w)

    du[3, 10] = σ * sqrt(τ * x₀)
    du[3, 11] = σ * sqrt(τ * k1 * v^2 * x)

    du[4, 12] = σ * sqrt(τ * k1 * v^2 * x)
    du[4, 13] = σ * sqrt(τ * k2 * (1/(v^3 + 1)) * y)

    du[5, 14] = σ * sqrt(τ * kcext * z)
    du[5, 15] = σ * sqrt(τ * kcext * Cmedia)
    du[5, 16] = σ * sqrt(τ * k2 * (1/(v^3 + 1)) * y)
    du[5, 17] = σ * sqrt(τ * v*z)
    du[5, 18] = σ * sqrt(τ * kz*exp(-D)*z2)
    du[5, 19] = σ * sqrt(τ * kz*exp(-D)*z)

    du[6, 20] = σ * sqrt(τ * kcext * z2)
    du[6, 21] = σ * sqrt(τ * kcext * Cmedia)
    du[6, 22] = σ * sqrt(τ * k2 * (1/(v2^3 + 1)) * y2)
    du[6, 23] = σ * sqrt(τ * v2*z2)
    du[6, 24] = σ * sqrt(τ * kz*exp(-D)*z)
    du[6, 25] = σ * sqrt(τ * kz*exp(-D)*z2)

    du[7, 26] = σ * sqrt(τ * v₀)
    du[7, 27] = σ * sqrt(τ * α*v2)
    du[7, 28] = σ * sqrt(τ * β*v2^2)
    du[7, 29] = σ * sqrt(τ * v2^3)
    du[7, 30] = σ * sqrt(τ * w2*v2)
    du[7, 31] = σ * sqrt(τ * v2*z2)

    du[8, 32] = σ * sqrt(τ * ε*w₀)
    du[8, 33] = σ * sqrt(τ * ε*γ*v2^2 / (v2^2 + k))
    du[8, 34] = σ * sqrt(τ * ε*w2)

    du[9, 35] = σ * sqrt(τ * x₀)
    du[9, 36] = σ * sqrt(τ * k1 * v2^2 * x2)

    du[10, 37] = σ * sqrt(τ * k1 * v2^2 * x2)
    du[10, 38] = σ * sqrt(τ * k2 * (1/(v2^3 + 1)) * y2)

end

# Integration time span
tspan = (0.0, 1000.0)

# Parameter values
τ = 19.5 / 2.3
ε = 0.55
α = 12.4
β = 8.05
γ = 8
k = 6
v₀ = 5.6
w₀ = 0.1
x₀ = 1.0
k1 = 0.1
k2 = 1.0
kz = 10.0
kcext = 5
Cmedia = 1
D = Inf
Ω = 1000
p = [τ, ε, α, β, γ, k, v₀, w₀, x₀, k1, k2, kz, kcext, Cmedia, D, Ω]

# Initial conditions
u0 = [0.7065, 0.7145, 0, 0, Cmedia, Cmedia, 0.7065, 0.7145, 0, 0]

# Construct the SDE problem
noise_rate_prototype = zeros(10, 38)
prob = SDEProblem(model!, noise!, u0, tspan, p; noise_rate_prototype=noise_rate_prototype)


## ----------------------------------------------------------------------------
## Plot example traces for default parameter values ---------------------------
## ----------------------------------------------------------------------------

# Simulate the model
sol = solve(prob, EM(), dt=0.001, saveat=900:0.01:950)
t = sol.t .- sol.t[1]  # so time starts at 0

# Create and save the figure
fig, ax = subplots(5, figsize=(6.5, 4.5), gridspec_kw=Dict("height_ratios" => [1, 1, 1, 1, 1.15]))
ax[1].plot(t, sol[1, :], color="black", label="A")
ax[1].set_ylabel("A\$_{1}\$", labelpad=0)
ax[1].set_title("Model components", pad=0, loc="left")
ax[2].plot(t, sol[2, :], color="black", label="I")
ax[2].set_ylabel("I\$_{1}\$", labelpad=0)
ax[3].plot(t, sol[3, :], color="black", label="X")
ax[3].set_ylabel("X\$_{1}\$", labelpad=0)
ax[4].plot(t, sol[4, :], color="black", label="Y")
ax[4].set_ylabel("Y\$_{1}\$", labelpad=0)
ax[5].plot(t, sol[5, :], color="black", label="Z")
ax[5].set_ylabel("Z\$_{1}\$", labelpad=0)
ax[5].set_xlabel("Time (min)", labelpad=0)
for i = 1:5
    ax[i].set_xlim(0, 20)
end
fig.tight_layout(pad=0)
fig.savefig("model_components.svg")


## ----------------------------------------------------------------------------
## Cmedia scan ----------------------------------------------------------------
## ----------------------------------------------------------------------------

# Scan for the amount of Cmedia in the extracellular space
Cmedia_arr = 0:0.1:2
amplitude_arr = Matrix{Float64}(undef, length(Cmedia_arr), 20)
progress_meter = Progress(length(Cmedia_arr), barlen=20)
Threads.@threads for i = 1:length(Cmedia_arr)

    global sol_min, sol_max
    local Cmedia

    Cmedia = Cmedia_arr[i]
    p2 = deepcopy(p)
    p2[end-2] = Cmedia
    prob2 = remake(prob, p=p2, tspan=tspan)
    for j = 1:10
        sol2 = solve(prob2, EM(), dt=0.001, saveat=200:0.1:tspan[end])
        amp1 = std(sol2[1, :])
        amp2 = std(sol2[7, :])
        amplitude_arr[i, j] = amp1
        amplitude_arr[i, j+10] = amp2
    end
    next!(progress_meter)

end

# Plot and save the results of the Cmedia scan
fig, ax = subplots(figsize=(6.5, 2))
ax.errorbar(Cmedia_arr,mean(amplitude_arr, dims=2)[:], yerr=std(amplitude_arr, dims=2)[:], fmt="o", color="black")
ax.set_xlabel("Ca\$^{2+}\$ medium", labelpad=0)
ax.set_ylabel("std(A\$_{1}\$)", labelpad=0)
ax.set_title("Ca\$^{2+}\$ scan", pad=0, loc="left")
ax.set_xlim(-0.05, 2.05)
fig.tight_layout(pad=0)
fig.savefig("Ca_medium_scan.svg")


## ----------------------------------------------------------------------------
## Scan distance --------------------------------------------------------------
## ----------------------------------------------------------------------------

distance_arr = range(0, 5, length=26)
n_repeat = 25
progress_meter = Progress(length(distance_arr), barlen=20)
R_arr = Matrix{Float64}(undef, length(distance_arr), n_repeat)
Threads.@threads for i_distance in 1:length(distance_arr)
    distance = distance_arr[i_distance]
    p2 = deepcopy(p)
    p2[end-1] = distance
    for i_repeat = 1:n_repeat
        prob2 = remake(prob, p=p2, tspan=(0.0, 1000))
        sol2 = solve(prob2, EM(), dt=0.001, saveat=300:0.01:1000)
        R = cor(sol2[1, :], sol2[7, :])
        R_arr[i_distance, i_repeat] = R
    end
    next!(progress_meter)
end

# Create and save figure for distance scan
fig, ax = subplots(figsize=(7, 1.5))

ax.errorbar(distance_arr, mean(R_arr, dims=2)[:], yerr=std(R_arr, dims=2)[:], fmt="o", color="black")
ax.set_xlabel("Distance", labelpad=0)
ax.set_ylabel("Correlation", labelpad=0)
ax.set_title("Decreasing cell distance", pad=0, loc="left")
ax.set_xlim(-0.1, 5.1)
ax.invert_xaxis()

fig.tight_layout(pad=0.3)
fig.savefig("distance_scan.svg")


## ----------------------------------------------------------------------------
## Example trace for short and long distance ----------------------------------
## ----------------------------------------------------------------------------

variable_pairs = [(1, 7, "A"), (2, 8, "I"), (3, 9, "X"), (4, 10, "Y"), (5, 6, "Z")]
n_variable_pairs = length(variable_pairs)
fig, ax_array = subplots(n_variable_pairs, 2, figsize=(7, 5))

# Simulate and plot short distance
p2 = deepcopy(p)
p2[end-1] = 0.0
prob2 = remake(prob, p=p2, tspan=tspan)
sol2 = solve(prob2, EM(), dt=0.001, saveat=980:0.01:1000)

for i = 1:n_variable_pairs

    local ax, t

    ax = ax_array[i, 1]
    variable_pair = variable_pairs[i]

    t = sol2.t .- sol2.t[1]
    ax.plot(t, sol2[variable_pair[1], :], label="Cell 1", color="red")
    ax.plot(t, sol2[variable_pair[2], :], label="Cell 2", color="blue")
    # ax.legend(ncol=2, edgecolor="black", framealpha=1.0)
    ax.set_xlim(0, 10)
    ax.set_ylabel(variable_pair[3], labelpad=0)

    if i == 1
        ax.set_title("Short distance", pad=0, loc="left")
    elseif i == n_variable_pairs
        ax.set_xlabel("Time (min)", labelpad=0)
    end

end

# Simulate and plot long distance
p2 = deepcopy(p)
p2[end-1] = 10
prob2 = remake(prob, p=p2, tspan=tspan)
sol2 = solve(prob2, EM(), dt=0.001, saveat=980:0.01:1000)

for i = 1:length(variable_pairs)

    ax = ax_array[i, 2]
    variable_pair = variable_pairs[i]

    t = sol2.t .- sol2.t[1]
    ax.plot(t, sol2[variable_pair[1], :], label="Cell 1", color="red")
    ax.plot(t, sol2[variable_pair[2], :], label="Cell 2", color="blue")
    ax.set_xlim(0, 10)
    ax.set_ylabel(variable_pair[3], labelpad=0)

    if i == 1
        ax.set_title("Long distance", pad=0, loc="left")
    elseif i == n_variable_pairs
        ax.set_xlabel("Time (min)", labelpad=0)
    end
    
end

# Save the figure
fig.tight_layout(pad=0.3)
fig.savefig("distance_example_traces.svg")


## ----------------------------------------------------------------------------
## Change distance during the simulation --------------------------------------
## ----------------------------------------------------------------------------

# Create a callback that will gradually decrease the distance parameter
condition = (u, t, integrator) -> true
affect! = function (integrator)
    t_start = 305
    t_end = 325
    D0 = 5
    t = integrator.t
    if t < t_start
        D = D0
    elseif t > t_end
        D = 0
    else
        D = D0 - (t - t_start) / (t_end - t_start) * D0
    end
    integrator.p[end-1] = D
end
callback = DiscreteCallback(condition, affect!, save_positions=(false, false))

# Create callback that will save the changing distance as a function of time
saved_values = SavedValues(Float64, Array{Float64})
saving_callback = SavingCallback((u, t, integrator) -> ([integrator.t, integrator.p[end-1]]), saved_values)

# Create callback set from the callbacks created above
callback_set = CallbackSet(callback, saving_callback)

# Simulate the model
prob2 = remake(prob, tspan=(0.0, 500.0), u0=rand(10))
sol = solve(prob2, EM(), dt=0.001, saveat=0:0.01:500, callback=callback_set)

# Get the changing value for the distance
t_D_array = [x[1] for x in saved_values.saveval]
D_array = [x[2] for x in saved_values.saveval]

# Create and save figure
fig, ax_array = subplots(2, 1, figsize=(7, 2), gridspec_kw=Dict("height_ratios" => [1, 2]))

t_offset = 300
y1 = sol[1, :]
y2 = sol[7, :]

ax = ax_array[1]
ax.plot(t_D_array .- t_offset, D_array, color="black")
ax.set_xlim(0, 30)
ax.set_ylabel("Distance", labelpad=0)
ax.set_title("Gradual distance decrease", pad=0, loc="left")

ax = ax_array[2]
ax.plot(sol.t .- t_offset, y1, color="blue", label="Cell 1")
ax.plot(sol.t .- t_offset, y2, color="red", label="Cell 2")

ax.set_xlim(0, 30)
ax.set_ylabel("A", labelpad=0)
ax.set_xlabel("Time (min)", labelpad=0)

fig.tight_layout(pad=0.3)
fig.savefig("changing_distance.svg")
