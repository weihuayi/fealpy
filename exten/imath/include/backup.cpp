
    void symmetric_strength_of_connection(CSRMatrix<I, T> & S, T theta)
    {
        std::vector<T> diags(shape[0]);

        //compute norm of diagonal values
        for(I i = 0; i < shape[0]; i++)
        {
            T diag = 0.0;
            for(I j = indptr[i]; j < indptr[i+1]; j++)
            {
                if(indices[j] == i)
                {
                    diag += data[j]; 
                }
            }
            diags[i] = std::abs(diag);
        }

        S.nnz = 0;
        S.indptr[0] = 0;

        for(I i = 0; i < shape[0]; i++)
        {
            T eps_Aii = theta*theta*diags[i];

            for(I jj = indptr[i]; jj < indptr[i+1]; jj++)
            {
                I j = indices[jj];
                if(i == j)
                {
                    // Always add the diagonal
                    S.indices[nnz] =   j;
                    S.data[nnz] = data[jj];
                    nnz++;
                }
                else if(data[jj]*data[jj] >= eps_Aii * diags[j]){
                    //  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|)
                    S.indices[nnz] =   j;
                    S.data[nnz] = data[jj];
                    nnz++;
                }
            }
            S.indptr[i+1] = nnz;
        }
    }

    I standard_aggregation(I x[],  I y[])
    {
        std::fill(x, x + shape[0], 0);

        I next_aggregate = 1; // number of aggregates + 1

        //Pass #1
        for(I i = 0; i < shape[0]; i++)
        {
            if(x[i]){ continue; } //already marked

            const I row_start = indptr[i];
            const I row_end   = indptr[i+1];

            //Determine whether all neighbors of this node are free (not already aggregates)
            bool has_aggregated_neighbors = false;
            bool has_neighbors            = false;
            for(I jj = row_start; jj < row_end; jj++)
            {
                const I j = indices[jj];
                if( i != j )
                {
                    has_neighbors = true;
                    if( x[j] )
                    {
                        has_aggregated_neighbors = true;
                        break;
                    }
                }
            }

            if(!has_neighbors)
            {
                //isolated node, do not aggregate
                x[i] = -shape[0];
            }
            else if (!has_aggregated_neighbors){
                //Make an aggregate out of this node and its neighbors
                x[i] = next_aggregate;
                y[next_aggregate-1] = i;              //y stores a list of the Cpts
                for(I jj = row_start; jj < row_end; jj++){
                    x[indices[jj]] = next_aggregate;
                }
                next_aggregate++;
            }
        }

        //Pass #2
        // Add unaggregated nodes to any neighboring aggregate
        for(I i = 0; i < shape[0]; i++)
        {
            if(x[i]){ continue; } //already marked

            for(I jj = indptr[i]; jj < indptr[i+1]; jj++)
            {
                const I j = indices[jj];
                const I xj = x[j];
                if(xj > 0)
                {
                    x[i] = -xj;
                    break;
                }
            }
        }

        next_aggregate--;

        //Pass #3
        for(I i = 0; i < shape[0]; i++)
        {
            const I xi = x[i];

            if(xi != 0)
            {
                // node i has been aggregated
                if(xi > 0)
                    x[i] = xi - 1;
                else if(xi == -shape[0])
                    x[i] = -1;
                else
                    x[i] = -xi - 1;
                continue;
            }

            // node i has not been aggregated
            const I row_start = indptr[i];
            const I row_end   = indptr[i+1];

            x[i] = next_aggregate;
            y[next_aggregate] = i;              //y stores a list of the Cpts

            for(I jj = row_start; jj < row_end; jj++)
            {
                const I j = indices[jj];
                if(x[j] == 0){ //unmarked neighbors
                    x[j] = next_aggregate;
                }
            }
            next_aggregate++;
        }


        return next_aggregate; //number of aggregates
    }
