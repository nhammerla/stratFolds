function stratBatchIter(targets, approxBatchsize)
	local T = targets:clone()+torch.randn(targets:size())*0.001 -- add some jitter for randomness
	local nbatch = torch.floor(T:size(1) / approxBatchsize)
    local batchIds = torch.zeros(T:size())

	-- index of sorted elements 
	local _, ind = torch.sort(T)

    -- prepare batch table
    batches = {}
    for i=1,nbatch do
        batches[i] = {}
    end

	
	-- assign a batch-id to each sample
    for i=1,ind:size(1) do
        -- assign elements in ascending order to batches
        -- ind[1] -> 1
        -- ind[2] -> 2 
        -- ...
        -- ind[nbatch]   -> nbatch
        -- ind[nbatch+1] -> 1
        -- ...
        table.insert(batches[1 + (i-1) % nbatch], ind[i])
        --batchIds[ind[i]] = 1 + (i-1) % nbatch
    end

	-- return iterator, which yields a bytetensor for addressing data
	local batchIndex = 0
	return function()
		batchIndex = batchIndex + 1
		if batchIndex <= nbatch then
            return torch.LongTensor(batches[batchIndex])
		    --return torch.eq(batchIds, batchIndex)
        end
	end
end
