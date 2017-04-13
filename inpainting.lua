require 'image'
require 'nn'
require 'optim'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 64,         -- size of a batch of images
    beta1 = 0.9,            -- beta param for adam
    dataset = 'folder',     -- folder (see data/donkey_folder.lua)
    display = 2929,
    gpu = 1,
    imgSize = 64,
    lambda = 0.002,
    loadSize = 64,
    lr = 0.02,
    name = 'celebA',
    net = 'celebA-normal',
    nIter = 2000,
    noise = 'normal',
    nz = 100,
    showEvery = 20,
    winId = 1000,
}

opt = xlua.envparams(opt)
print(opt)

if opt.display > 0 then
    display = require 'display'
    display.configure({ hostname = '0.0.0.0', port = opt.display })
else
    opt.display = false
end

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.gpu)
end

optimConfig = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}

-- load the networks
local netD = 'checkpoints/' .. opt.net .. '_net_D.t7'
local netG = 'checkpoints/' .. opt.net .. '_net_G.t7'
netG = torch.load(netG)
netD = torch.load(netD)

-- local L1Criterion = nn.AbsCriterion()
local L1Criterion = nn.SmoothL1Criterion()
local L2Criterion = nn.MSECriterion()
local BCECriterion = nn.BCECriterion()

-- get a ramdom batch of images to complete
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(0, opt.dataset, opt)
local images = data:getBatch()

-- mask the images
local height = images:size(3)
local width = images:size(4)
mask = torch.Tensor(images:size()):fill(1)
mask:narrow(3, height / 4, height / 2):narrow(4, width / 4, width / 2):zero()

-- sample some noise
z = torch.Tensor(images:size(1), opt.nz, 1, 1)
if opt.noise == 'uniform' then
    z:uniform(-1, 1)
else
    z:normal(0, 1)
end

local label = torch.Tensor(images:size(1)):fill(1)

if opt.gpu > 0 then
    netD:cuda()
    netG:cuda()
    cudnn.convert(netD, cudnn)
    cudnn.convert(netG, cudnn)
    L1Criterion:cuda()
    L2Criterion:cuda()
    BCECriterion:cuda()
    mask = mask:cuda()
    images = images:cuda()
    z = z:cuda()
    label = label:cuda()
end

local masked_img = torch.cmul(images, mask)

-- function performing completion
local complete = function(masked_img, mask, z)
    local gen = netG:forward(z)
    return masked_img + torch.cmul(mask:clone():fill(1) - mask, gen)
end

-- closure computing df/dz and f(z)
local loss_dL_dz = function(z)
    mlpG = netG:clone('weight', 'bias');
    mlpD = netD:clone('weight', 'bias');

    -- contextual loss
    local gen = mlpG:forward(z)
    local contextual_err = L1Criterion:forward(torch.cmul(gen, mask), masked_img)
    local df_do_con = L1Criterion:backward(torch.cmul(gen, mask), masked_img)

    -- perceptual loss
    local pred = mlpD:forward(gen)
    local perceptual_err = BCECriterion:forward(pred, label)
    local df_do_per = BCECriterion:backward(pred, label)
    local dD_dz = mlpD:updateGradInput(gen, df_do_per)

    local grads = mlpG:updateGradInput(z, torch.cmul(df_do_con, mask) + opt.lambda * dD_dz)

    -- sum
    local err = contextual_err + opt.lambda * perceptual_err
    -- print(err, contextual_err, perceptual_err)

    return err, grads
end

print 'Inpainting...'

local save_dir = 'completed/' .. opt.name
paths.mkdir(save_dir)

if display then
    display.image(images, { win = opt.winId, title = "original images" })
    display.image(masked_img, { win = opt.winId + 1, title = "masked images" })
    display.image(complete(masked_img, mask, z), { win = opt.winId + 2, title = "inpainted images" })
end

image.save(save_dir .. '/image_000.jpg', image.toDisplayTensor({ input = masked_img, nrow = 8 }))

for iter = 1, opt.nIter do
    z = optim.adam(loss_dL_dz, z, optimConfig)

    if opt.noise == 'uniform' then z = z:clamp(-1, 1) end

    -- display in browser and save images
    if iter % opt.showEvery == 0 then
        local gen = netG:forward(z)
        local masked_gen = torch.cmul(gen, mask)
        local comp_img = complete(masked_img, mask, z)
        if display then
            display.image(comp_img, { win = opt.winId + 2, title = "inpainted images" })
            display.image(masked_gen, { win = opt.winId + 4, title = "masked generated images" })
            display.image(gen, { win = opt.winId + 3, title = "generated images" })
        end

        image.save((save_dir .. '/image_%03d.jpg'):format(iter / opt.showEvery), image.toDisplayTensor({ input = comp_img, nrow = 8 }))

        xlua.progress(iter / opt.showEvery, opt.nIter / opt.showEvery)
    end
end

print("Completed images have been saved into " .. save_dir .. ".")
