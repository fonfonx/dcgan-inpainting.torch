require 'image'
require 'nn'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 64,
    beta1 = 0.9,
    dataset = 'folder',
    display = 2929,
    imgSize = 64,
    gpu = 1,
    lambda = 0.004,
    loadSize = 64,
    lr = 0.03,
    netD = 'checkpoints/celebA-soumith_9_net_D.t7',
    netG = 'checkpoints/celebA-soumith_9_net_G.t7',
    noise = 'normal'
    nIter = 10000,
    win_id = 1000,
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.display > 0 then
    display = require 'display'
    display.configure({ hostname = '0.0.0.0', port = opt.display })
end

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

optimConfig = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}

netG = torch.load(opt.netG)
netD = torch.load(opt.netD)

-- local L1Criterion = nn.AbsCriterion()
local L1Criterion = nn.SmoothL1Criterion()
local L2Criterion = nn.MSECriterion()
local BCECriterion = nn.BCECriterion()

img = image.load('celebA/images/000008.jpg')

local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(0, opt.dataset, opt)
local images = data:getBatch()

local height = images:size(3)
local width = images:size(4)

mask = torch.Tensor(images:size()):fill(1)
mask:narrow(3, 16, 32):narrow(4, 16, 32):zero()

z = torch.Tensor(images:size(1), 100, 1, 1)
if noise == 'uniform' then
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

local complete = function(masked_img, mask, z)
    local gen = netG:forward(z)
    return masked_img + torch.cmul(mask:clone():fill(1) - mask, gen)
end

if display then
    display.image(images, { win = opt.win_id, title = "original images" })
    display.image(masked_img, { win = opt.win_id + 1, title = "masked images" })
    display.image(complete(masked_img, mask, z), { win = opt.win_id + 2, title = "inpainted images" })
end

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

for iter = 1, opt.nIter do
    z = optim.adam(loss_dL_dz, z, optimConfig):clamp(-1, 1)
    if iter % 20 == 0 then
        local gen = netG:forward(z)
        local masked_gen = torch.cmul(gen, mask)
        local comp_img = complete(masked_img, mask, z)
        display.image(comp_img, { win = opt.win_id + 2, title = "inpainted images" })
        display.image(masked_gen, { win = opt.win_id + 4, title = "masked generated images" })
        display.image(gen, { win = opt.win_id + 3, title = "generated images" })
    end
end
