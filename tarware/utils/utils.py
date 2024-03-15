def flatten_list(l):
    '''
    Flattens a list of lists
    '''
    return [item for sublist in l for item in sublist]

def split_list(lst, n_groups, verbose=False):
    '''
    Splits a list `lst` into `n_groups` groups of approximately equal lengths. 
    This function guarantees that the chunks will be the same length, or differing in length by 1 element at most.
    '''
    # If you divide n elements into roughly k chunks you can make n % k chunks 1 element bigger than the other chunks to distribute the extra elements.
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(lst), n_groups)
    output = list(lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n_groups))
    
    if verbose:
        group_lengths = [len(group) for group in output]
        min_group_length = min(group_lengths)
        max_group_length = max(group_lengths)
        num_groups_with_min_group_length = len([i for i in group_lengths if i == min_group_length])
        num_groups_with_max_group_length = len([i for i in group_lengths if i == max_group_length])

        print(f"Splitting {len(lst)} items into {n_groups} groups")

        if min_group_length == max_group_length:
            print("Even split achieved")
            print(f"Group size = {min_group_length} ({num_groups_with_min_group_length} groups total)")
        else:
            print("Non-even split achieved")
            print(f"Minimum group size = {min_group_length} ({num_groups_with_min_group_length} groups total)")
            print(f"Maximum group size = {max_group_length} ({num_groups_with_max_group_length} groups total)")
    return output

def find_sections(pairs):
    groups = []

    for pair in pairs:
        added = False

        for group in groups:
            if any(abs(pair[0] - gp[0]) + abs(pair[1] - gp[1]) == 1 for gp in group):
                group.append(pair)
                added = True
                break

        if not added:
            groups.append([pair])

    return groups