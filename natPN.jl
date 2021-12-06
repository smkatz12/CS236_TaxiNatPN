using Flux
using Bijectors

function build_model(layer_sizes, act)
    # ReLU except last layer identity
    layers = Any[Dense(layer_sizes[i], layer_sizes[i+1], act) for i = 1:length(layer_sizes)-2]
    push!(layers, Dense(layer_sizes[end-1], layer_sizes[end]))
    return Chain(layers...)
end

function build_radial_flow(nlayers, dim; device = cpu)
    d = MvNormal(zeros(dim) |> device, ones(dim) |> device)

    b = RadialLayer(dim) |> device
    for i = 2:nlayers
        b = b ∘ RadialLayer(dim)
    end

    return transformed(d, b)
end

mutable struct NatPN
    encoder
    flow
    decoder
    h # latent dimension
    Nₕ # certainty budget
    entropy_reg # entropy tradeoff in loss
    σ² # aleotoric uncertainty
end

Flux.trainable(npn::NatPN) = (npn.encoder, npn.flow, npn.decoder)

function natPN(; input_dim = 128,
    encoder_hidden_layers = [64, 32, 16],
    nflow_layers = 8,
    h = 4,
    entropy_reg = 1e-5,
    Nₕ = 0.5 * (h * log(2π) + log(h + 1)),
    σ² = 0.01^2,
    device = cpu)

    encoder = build_model([input_dim; encoder_hidden_layers; h], relu) |> device

    flow = build_radial_flow(nflow_layers, h) |> device

    decoder = Dense(h, 1) |> device

    return NatPN(encoder, flow, decoder, h, Nₕ, entropy_reg, σ²)
end

function natPN_2d(; input_dim = 128,
    encoder_hidden_layers = [64, 32, 16],
    nflow_layers = 8,
    h = 4,
    entropy_reg = 1e-5,
    Nₕ = 0.5 * (h * log(2π) + log(h + 1)),
    σ² = 0.01^2,
    device = cpu)

    encoder = build_model([input_dim; encoder_hidden_layers; h], relu) |> device

    flow = build_radial_flow(nflow_layers, h) |> device

    decoder = Dense(h, 2) |> device

    return NatPN(encoder, flow, decoder, h, Nₕ, entropy_reg, σ²)
end

function natPN_nonlinear(; input_dim = 128,
    encoder_hidden_layers = [64, 32, 16],
    decoder_hidden_layers = [16, 16, 8, 4],
    nflow_layers = 8,
    h = 4,
    entropy_reg = 1e-5,
    Nₕ = 0.5 * (h * log(2π) + log(h + 1)),
    σ² = 0.01^2,
    device = cpu)

    encoder = build_model([input_dim; encoder_hidden_layers; h], relu) |> device

    flow = build_radial_flow(nflow_layers, h) |> device

    decoder = build_model([h; decoder_hidden_layers; 1], relu) |> device

    return NatPN(encoder, flow, decoder, h, Nₕ, entropy_reg, σ²)
end

function natPN_nonlinear_2d(; input_dim = 128,
    encoder_hidden_layers = [64, 32, 16],
    decoder_hidden_layers = [16, 16, 8, 4],
    nflow_layers = 8,
    h = 4,
    entropy_reg = 1e-5,
    Nₕ = 0.5 * (h * log(2π) + log(h + 1)),
    σ² = 0.01^2,
    device = cpu)

    encoder = build_model([input_dim; encoder_hidden_layers; h], relu) |> device

    flow = build_radial_flow(nflow_layers, h) |> device

    decoder = build_model([h; decoder_hidden_layers; 2], relu) |> device

    return NatPN(encoder, flow, decoder, h, Nₕ, entropy_reg, σ²)
end

function bayes_loss(m::NatPN, x, y)
    """ Assumes x is of size (xdim, batchsize)
    note! not currently compatible with 2D
    """
    z = m.encoder(x)
    pz = exp.(logpdf(m.flow, z))

    χ = m.decoder(z)
    n = m.Nₕ * pz

    # Convert to normal
    μ₀ = χ * m.σ²
    σ₀² = m.σ² / n

    likelihood = -(1 ./ (2 * m.σ²)) .* sum((y .- μ₀).^2, dims = 1) .- (σ₀² ./ (2 .* m.σ²)) .- log(√(2π * m.σ²))
    entropy = (1 / 2) .* log.(2π .* σ₀²)

    if any(isnan.(entropy))
        println(minimum(σ₀²))
    end

    return sum(-likelihood .- m.entropy_reg .* entropy) / size(x, 2)
end

function bayes_loss_simple(m::NatPN, x, y; verbose = false)
    """ Assumes x is of size (xdim, batchsize)
    """
    z = m.encoder(x)
    pz = exp.(logpdf(m.flow, z))

    # Get distribution parameters
    μ₀ = m.decoder(z)
    n = m.Nₕ * pz

    likelihood_term = (1 / m.σ²) .* sum((y .- μ₀).^2, dims = 1) .+ (1 ./ n)
    entropy_term = log.(n)

    if verbose
        println("mse: ", sum((y .- μ₀) .^ 2) / size(x, 2))
        println("1 / n: ", sum(1 ./ n) / size(x, 2))
        println("entropy reg term: ", sum(m.entropy_reg .* entropy_term) / size(x, 2))
    end

    return -sum(-likelihood_term .- m.entropy_reg .* entropy_term) / size(x, 2)
end


function regression_loss(m::NatPN, x, y)
    z = m.encoder(x)
    w = m.decoder(z)

    return Flux.mse(w[1, :]', y)
end

function flow_loss(m::NatPN, x, y)
    z = m.encoder(x)
    return -sum(logpdf(m.flow, z)) / size(x, 2)
end

function to_normal(χ, n, σ²)
    μ₀ = χ * σ²
    σ₀² = σ² / n
    return μ₀, σ₀²
end

function get_posterior(m::NatPN, x; χ_prior = 0.0, n_prior = 1.0)
    """ Note! not currently compatible with 2D
    """
    z = m.encoder(x)
    pz = exp(logpdf(m.flow, z))
    χ = m.decoder(z)

    # Get χ and n
    n = m.Nₕ * pz
    χ = χ[1]

    # Posterior update
    χ_post = (n_prior * χ_prior + n * χ) / (n_prior + n)
    n_post = n_prior + n

    return χ_post, n_post
end

function get_posterior_simple(m::NatPN, x; χ_prior = 0.0, n_prior = 1.0)
    z = m.encoder(x)
    pz = exp(logpdf(m.flow, z))
    μ₀ = m.decoder(z)

    # Get χ and n
    n = m.Nₕ * pz
    χ = vec(μ₀) / m.σ²

    # Posterior update
    χ_post = (n_prior * χ_prior + n * χ) / (n_prior + n)
    n_post = n_prior + n

    return χ_post, n_post
end