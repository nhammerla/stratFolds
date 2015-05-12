function stratBatchIter(targets, approxBatchsize)
    -- clone the targets and add some jitter
	local T = targets:clone()+torch.randn(targets:size())*0.001
    -- we use floor here, batches will be a bit bigger than requested
	local nbatch = torch.floor(T:size(1) / approxBatchsize)

	-- indeces of sorted elements 
	local _, ind = torch.sort(T)

    -- prepare table for batches
    batches = {}
    for i=1,nbatch do
        batches[i] = {}
    end

	-- assign a batch-id to each sample
    for i=1,ind:size(1) do
        -- assign elements in ascending order to batches
        table.insert(batches[1 + (i-1) % nbatch], ind[i])
    end

	-- return iterator, which yields a list of indeces for each batch
	local batchIndex = 0
	return function()
		batchIndex = batchIndex + 1
		if batchIndex <= nbatch then
            return torch.LongTensor(batches[batchIndex])
		    --return torch.eq(batchIds, batchIndex)
        end
	end
end
