    conflicts = []
    train_idxes = []
    for _ in range(100):
        target_data = load_processed_data(opt, opt.data_path, opt.data_name,
                                          shuffle_seed=opt.shuffle_seed_list[iter_split_seed],
                                          ppr_file=opt.ppr_file)
        a, b = get_renode_weight(opt, target_data)
        conflicts.append(torch.sum(b[target_data.train_mask]))
        train_idxes.append(target_data.train_mask)
        iter_split_seed += 1
    sorted_conflicts = sorted(conflicts)
    for i in range(10):
        con = sorted_conflicts[i]
        idx = conflicts.index(con)
        name = str(i)+".pt"
        torch.save(train_idxes[idx], name)
    for i in range(45,55):
        con = sorted_conflicts[i]
        idx = conflicts.index(con)
        name = str(i)+".pt"
        torch.save(train_idxes[idx], name)
    for i in range(90,100):
        con = sorted_conflicts[i]
        idx = conflicts.index(con)
        name = str(i)+".pt"
        torch.save(train_idxes[idx], name)