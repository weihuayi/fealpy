#ifndef split_h
#define split_h


template<class SM>
void rs_cf_splitting(SM & S,  I splitting[], const int splitting_size)
{
    std::vector<I> lambda(n_nodes,0);

    //compute lambdas
    for(I i = 0; i < n_nodes; i++){
        lambda[i] = Tp[i+1] - Tp[i];
    }

    //for each value of lambda, create an interval of nodes with that value
    // ptr - is the first index of the interval
    // count - is the number of indices in that interval
    // index to node - the node located at a given index
    // node to index - the index of a given node
    std::vector<I> interval_ptr(n_nodes+1,0);
    std::vector<I> interval_count(n_nodes+1,0);
    std::vector<I> index_to_node(n_nodes);
    std::vector<I> node_to_index(n_nodes);

    for(I i = 0; i < n_nodes; i++){
        interval_count[lambda[i]]++;
    }
    for(I i = 0, cumsum = 0; i < n_nodes; i++){
        interval_ptr[i] = cumsum;
        cumsum += interval_count[i];
        interval_count[i] = 0;
    }
    for(I i = 0; i < n_nodes; i++){
        I lambda_i = lambda[i];
        I index    = interval_ptr[lambda_i] + interval_count[lambda_i];
        index_to_node[index] = i;
        node_to_index[i]     = index;
        interval_count[lambda_i]++;
    }


    std::fill(splitting, splitting + n_nodes, U_NODE);

    // all nodes with no neighbors become F nodes
    for(I i = 0; i < n_nodes; i++){
        if (lambda[i] == 0 || (lambda[i] == 1 && Tj[Tp[i]] == i))
            splitting[i] = F_NODE;
    }

    //Now add elements to C and F, in descending order of lambda
    for(I top_index = n_nodes - 1; top_index != -1; top_index--){
        I i        = index_to_node[top_index];
        I lambda_i = lambda[i];

        //if (n_nodes == 4)
        //    std::cout << "selecting node #" << i << " with lambda " << lambda[i] << std::endl;

        //remove i from its interval
        interval_count[lambda_i]--;

        if(splitting[i] == F_NODE)
        {
            continue;
        }
        else
        {
            assert(splitting[i] == U_NODE);

            splitting[i] = C_NODE;

            //For each j in S^T_i /\ U
            for(I jj = Tp[i]; jj < Tp[i+1]; jj++){
                I j = Tj[jj];

                if(splitting[j] == U_NODE){
                    splitting[j] = F_NODE;

                    //For each k in S_j /\ U
                    for(I kk = Sp[j]; kk < Sp[j+1]; kk++){
                        I k = Sj[kk];

                        if(splitting[k] == U_NODE){
                            //move k to the end of its current interval
                            if(lambda[k] >= n_nodes - 1) continue;

                            I lambda_k = lambda[k];
                            I old_pos  = node_to_index[k];
                            I new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k] - 1;

                            node_to_index[index_to_node[old_pos]] = new_pos;
                            node_to_index[index_to_node[new_pos]] = old_pos;
                            std::swap(index_to_node[old_pos], index_to_node[new_pos]);

                            //update intervals
                            interval_count[lambda_k]   -= 1;
                            interval_count[lambda_k+1] += 1; //invalid write!
                            interval_ptr[lambda_k+1]    = new_pos;

                            //increment lambda_k
                            lambda[k]++;
                        }
                    }
                }
            }

            //For each j in S_i /\ U
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                I j = Sj[jj];
                if(splitting[j] == U_NODE){            //decrement lambda for node j
                    if(lambda[j] == 0) continue;

                    //assert(lambda[j] > 0);//this would cause problems!

                    //move j to the beginning of its current interval
                    I lambda_j = lambda[j];
                    I old_pos  = node_to_index[j];
                    I new_pos  = interval_ptr[lambda_j];

                    node_to_index[index_to_node[old_pos]] = new_pos;
                    node_to_index[index_to_node[new_pos]] = old_pos;
                    std::swap(index_to_node[old_pos],index_to_node[new_pos]);

                    //update intervals
                    interval_count[lambda_j]   -= 1;
                    interval_count[lambda_j-1] += 1;
                    interval_ptr[lambda_j]     += 1;
                    interval_ptr[lambda_j-1]    = interval_ptr[lambda_j] - interval_count[lambda_j-1];

                    //decrement lambda_j
                    lambda[j]--;
                }
            }
        }
    }
}

#endif // end of split_h
