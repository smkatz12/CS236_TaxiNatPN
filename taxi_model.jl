using Flux.Data
using Flux.Optimise
using HDF5
using Random

include("natPN.jl")

function get_taxi_data(yinds;
    fn = "/home/smkatz/Documents/NASA_ULI/ImageVerif/VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5",
    flipped = true)

    images = flipped ? h5read(fn, "y_train") : h5read(fn, "X_train")
    X = reshape(images, 128, :)
    y = flipped ? h5read(fn, "X_train") ./ [10.0, 30.0, 1.0] : h5read(fn, "y_train") ./ [10.0, 30.0, 1.0]
    y = y[yinds, :]
    if length(yinds) == 1
        return X, y'
    else
        return X, y
    end
end

function train_npn(npn, X, y, nepoch, lr, batchsize; simple = false)
    # Set up the data
    test_size = 512
    train_range = 1:size(X, 2)-test_size
    test_range = train_range.stop+1:size(X, 2)
    dtrain = DataLoader((X[:, train_range], y[:, train_range]),
        batchsize = batchsize, shuffle = true)
    test_x, test_y = (X[:, test_range], y[:, test_range])

    # Set up loss
    bayes_loss_train(x, y) = simple ? bayes_loss_simple(npn, x, y) : bayes_loss(npn, x, y)
    bayes_loss_display(x, y) = simple ? bayes_loss_simple(npn, x, y, verbose = true) : bayes_loss(npn, x, y)

    # Callback to track progress
    function evalcb()
        l = bayes_loss_display(test_x, test_y)
        @show l
    end

    # Set up optimizer
    opt = ADAM(lr)

    # Train
    Flux.@epochs nepoch Flux.train!(bayes_loss_train, Flux.params(npn), dtrain, opt, cb = Flux.throttle(evalcb, 20))

    return npn
end

function train_npn_fine_tune(npn, X, y; lr_reg = 1e-4, lr_flow = 1e-4, lr_bayes = 1e-4,
    batchsize = 128,
    nreg = 50, nflow = 50, nbayes = 50)
    # Set up the data
    test_size = 512
    train_range = 1:size(X, 2)-test_size
    test_range = train_range.stop+1:size(X, 2)
    dtrain = DataLoader((X[:, train_range], y[:, train_range]),
        batchsize = batchsize, shuffle = true)
    test_x, test_y = (X[:, test_range], y[:, test_range])

    # Set up losses
    reg_loss_train(x, y) = regression_loss(npn, x, y)
    flow_loss_train(x, y) = flow_loss(npn, x, y)
    bayes_loss_train(x, y) = bayes_loss(npn, x, y)

    # Callback to track progress
    function evalcb_reg()
        l = reg_loss_train(test_x, test_y)
        @show l
    end

    # Callback to track progress
    function evalcb_flow()
        l = flow_loss_train(test_x, test_y)
        @show l
    end

    # Callback to track progress
    function evalcb_bayes()
        l = bayes_loss_train(test_x, test_y)
        @show l
    end

    # Set up optimizer
    opt_reg = ADAM(lr_reg)
    opt_flow = ADAM(lr_flow)
    opt_bayes = ADAM(lr_bayes)

    # Train
    Flux.@epochs nreg Flux.train!(reg_loss_train, Flux.params(npn), dtrain, opt_reg, cb = Flux.throttle(evalcb_reg, 5))
    Flux.@epochs nflow Flux.train!(flow_loss_train, Flux.params(npn.flow), dtrain, opt_flow, cb = Flux.throttle(evalcb_flow, 5))
    Flux.@epochs nbayes Flux.train!(bayes_loss_train, Flux.params(npn), dtrain, opt_bayes, cb = Flux.throttle(evalcb_bayes, 5))

    return npn
end

function train_flow(flow, Z, y, nepoch, lr, batchsize)
    # Training data loader
    dtrain = DataLoader((Z, y), batchsize = batchsize, shuffle = true)

    # Get a test set just for displaying loss
    test_size = 512
    test_inds = randperm(size(Z, 2))[1:test_size]
    test_x, test_y = (Z[:, test_inds], Z[:, test_inds])

    # Set up loss
    function loss(x, y)
        return -sum(logpdf(flow, x)) / size(x, 2)
    end

    # Callback to track progress
    function evalcb()
        l = loss(test_x, test_y)
        @show l
    end

    # Set up optimizer
    opt = ADAM(lr)

    # Train
    Flux.@epochs nepoch Flux.train!(loss, Flux.params(flow), dtrain, opt, cb = Flux.throttle(evalcb, 20))

    return flow
end

function train_flow_ood(flow, Z, Zood, nepoch, lr, batchsize)
    # Training data loader
    dtrain = DataLoader((Z, Zood), batchsize = batchsize, shuffle = true)

    # Get a test set just for displaying loss
    test_size = 512
    test_inds = randperm(size(Z, 2))[1:test_size]
    test_x, test_y = (Z[:, test_inds], Zood[:, test_inds])

    # Set up loss
    function loss(x, y)
        return (-sum(logpdf(flow, x)) + sum(logpdf(flow, y))) / (2 * size(x, 2))
    end

    # Callback to track progress
    function evalcb()
        l = loss(test_x, test_y)
        @show l
    end

    # Set up optimizer
    opt = ADAM(lr)

    # Train
    Flux.@epochs nepoch Flux.train!(loss, Flux.params(flow), dtrain, opt, cb = Flux.throttle(evalcb, 20))

    return flow
end

function train_reconstruction(encoder, X; layer_sizes = [16, 16, 32, 64, 128], device = cpu,
    nepoch = 100, lr = 1e-3, batchsize = 128)

    decoder = build_model(layer_sizes, relu) |> device

    # Training data loader
    dtrain = DataLoader((X, X), batchsize = batchsize, shuffle = true)

    # Get a test set just for displaying loss
    test_size = 512
    test_inds = randperm(size(X, 2))[1:test_size]
    test_x, test_y = (X[:, test_inds], X[:, test_inds])

    # Set up loss
    loss(x, y) = Flux.mse(decoder(encoder(x)), y)

    # Callback to track progress
    function evalcb()
        l = loss(test_x, test_y)
        @show l
    end

    # Set up optimizer
    opt = ADAM(lr)

    # Train
    Flux.@epochs nepoch Flux.train!(loss, Flux.params(decoder, encoder),
        dtrain, opt, cb = Flux.throttle(evalcb, 20))

    return encoder
end

function train_decoder(decoder, z, y; nepoch = 100, lr = 1e-3, batchsize = 128)

    # Training data loader
    dtrain = DataLoader((z, y), batchsize = batchsize, shuffle = true)

    # Get a test set just for displaying loss
    test_size = 512
    test_inds = randperm(size(X, 2))[1:test_size]
    test_x, test_y = (z[:, test_inds], y[:, test_inds])

    # Set up loss
    loss(x, y) = Flux.mse(decoder(x), y)

    # Callback to track progress
    function evalcb()
        l = loss(test_x, test_y)
        @show l
    end

    # Set up optimizer
    opt = ADAM(lr)

    # Train
    Flux.@epochs nepoch Flux.train!(loss, Flux.params(decoder),
        dtrain, opt, cb = Flux.throttle(evalcb, 20))

    return decoder
end
