require 'image'
require 'nn'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    beta1 = 0.9,
    display = 2929,
    gpu = 1,
    lambda = 0.005,
    lr = 0.03,
    netD = 'checkpoints/celebA-raw-sigmoid_46_net_D.t7',
    netG = 'checkpoints/celebA-raw-sigmoid_46_net_G.t7',
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

-- img = image.load('lfw64/images/George_W_Bush_0129.jpg', 3)
img = image.load('celebA/images/000008.jpg')
-- img:mul(2):add(-1)

local height = img:size(2)
local width = img:size(3)

img = img:view(-1, 3, height, width)

mask = torch.Tensor(img:size()):fill(1)
mask2 = torch.Tensor(img:size()):fill(0)
mask2:narrow(3, 12, 40):narrow(4, 12, 40):fill(1)
mask:narrow(3, 16, 32):narrow(4, 16, 32):zero()
mask2:narrow(3, 16, 32):narrow(4, 16, 32):zero()

z = torch.Tensor(1, 100, 1, 1)
z:uniform(-1, 1)

local label = torch.Tensor(1):fill(1)

if opt.gpu > 0 then
    L1Criterion:cuda()
    L2Criterion:cuda()
    BCECriterion:cuda()
    mask = mask:cuda()
    mask2 = mask2:cuda()
    img = img:cuda()
    z = z:cuda()
    label = label:cuda()
end

local masked_img = torch.cmul(img, mask)
local masked_img2 = torch.cmul(img, mask2)

local complete = function(masked_img, mask, z)
    local gen = netG:forward(z)
    return masked_img + torch.cmul(mask:clone():fill(1) - mask, gen)
end

if display then
    display.image(img, { win = opt.win_id, title = "original image" })
    display.image(masked_img, { win = opt.win_id + 1, title = "masked image" })
    display.image(complete(masked_img, mask, z), { win = opt.win_id + 2, title = "inpainted image" })
end

local loss_dL_dz = function(z)
    mlpG = netG:clone('weight', 'bias');
    mlpD = netD:clone('weight', 'bias');
    -- contextual loss
    local gen = mlpG:forward(z)
    -- local contextual_err = L1Criterion:forward(gen, img)
    -- local df_do_con = L1Criterion:backward(gen, img)
    local contextual_err = L1Criterion:forward(torch.cmul(gen, mask), masked_img)
    local df_do_con = L1Criterion:backward(torch.cmul(gen, mask), masked_img)

    -- local dcon_dz = mlpG1:updateGradInput(z, df_do_con)
    -- perceptual loss
    local pred = mlpD:forward(gen)
    local perceptual_err = BCECriterion:forward(pred, label)
    local df_do_per = BCECriterion:backward(pred, label)
    local dD_dz = mlpD:updateGradInput(gen, df_do_per)

    local grads = mlpG:updateGradInput(z, torch.cmul(df_do_con, mask) + opt.lambda * dD_dz)
    -- -- print("p", perceptual_err)

    -- sum
    local err = contextual_err + opt.lambda * perceptual_err
    print(err, contextual_err, perceptual_err)
    -- print(grads)
    return err, grads
end

for iter = 1, opt.nIter do
    z = optim.adam(loss_dL_dz, z, optimConfig):clamp(-1, 1)
    -- print(z)
    print(z[1][1][1][1])
    if iter % 20 == 0 then
        local gen = netG:forward(z)--:clamp(0, 1)
        local masked_gen = torch.cmul(gen, mask)
        local comp_img = complete(masked_img, mask, z)
        display.image(comp_img, { win = opt.win_id + 2, title = "inpainted image" })
        display.image(masked_gen, { win = opt.win_id + 4, title = "masked gen" })
        display.image(gen, { win = opt.win_id + 3, title = "gen image" })
        -- print(masked_gen[1][30][30])
        -- print((masked_gen - masked_img):abs():narrow(2, 12, 40):narrow(3, 12, 40))
    end
end
