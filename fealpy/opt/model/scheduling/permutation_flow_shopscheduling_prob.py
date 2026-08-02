from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt

class PermutationFlowShopschedulingProb:
    """
    A class representing a permutation flow shop scheduling problem.

    This class encapsulates the evaluation, decoding, and visualization utilities
    for permutation flow shop scheduling (PFSP), where a sequence vector is
    mapped to a job order processed across multiple machines. The class handles
    makespan computation, schedule construction, and Gantt-chart-like visualization.

    Parameters:
        options(dict): A dictionary containing problem-specific data.
            Required field:
                - 'data' (Tensor): A (mach_qty, job_qty) processing-time matrix,
                  where each entry denotes the processing time of a job on a machine.

    Attributes:
        data(Tensor): Processing-time matrix used for evaluation.
        process_time(Tensor): A job-major processing-time matrix assigned during decoding.
        time(Tensor): Completion-time matrix after decoding a sequence.
        ctime(float): Final completion time (makespan) after decoding.
    """

    def __init__(self, options):
        """
        Initializes a new permutation flow shop scheduling instance.

        Parameters:
            options(dict): A configuration dictionary that must contain:
                - data (Tensor): Processing-time matrix with shape
                  (machine_quantity, job_quantity).
        """
        self.data = options['data']
    
    def get_bounds(self):
        """
        Returns the lower and upper bounds for sampling decision variables.

        This is typically used by optimization algorithms that require
        bounds on the continuous representation of a job sequence.

        Returns:
            tuple[int, int]: A pair (lower_bound, upper_bound) indicating the
            allowable numerical range for encoded job positions.
        """
        return 6, 200
    
    def evaluate(self, x):
        """
        Evaluates a batch of encoded job sequences and computes their makespans.

        Each row of `x` is interpreted as a continuous vector whose argsort
        determines the job processing order. A complete deterministic schedule
        is generated using classical flow shop rules: each job is processed on
        all machines in fixed order, and each machine handles jobs sequentially.

        Parameters:
            x(Tensor): A tensor of shape (n, job_qty), where each row is a
                continuous encoding of a job sequence. The decoding step uses
                argsort to obtain the processing order.

        Returns:
            Tensor: A tensor of shape (n,) containing the makespan (final
            completion time) for each decoded job sequence.
        """
        seq = bm.argsort(x)                       # shape: (n, job_qty)
        n = x.shape[0]
        mach_qty, job_qty = self.data.shape

        # Schedule table: [job, machine, info-fields]
        schedule = bm.zeros((n, mach_qty * job_qty, 5))

        # Track next available start time for each job / machine
        job_can_st = bm.zeros((n, job_qty))
        mach_can_st = bm.zeros((n, mach_qty))

        row = 0
        batch_index = bm.arange(n)

        # Iterate over job positions in the sequence
        for pos in range(job_qty):
            now_job = seq[:, pos]  # shape: (n,)

            for m in range(mach_qty):
                # Earliest start = max(job's availability, machine's availability)
                start_t = bm.maximum(
                    job_can_st[batch_index, now_job],
                    mach_can_st[:, m]
                )
                duration = self.data[m, now_job]
                end_t = start_t + duration

                # Schedule entry: (job_id, machine, machine, start, end)
                schedule[:, row, 0] = now_job
                schedule[:, row, 1] = m
                schedule[:, row, 2] = m
                schedule[:, row, 3] = start_t
                schedule[:, row, 4] = end_t

                # Update availability
                job_can_st[batch_index, now_job] = end_t
                mach_can_st[:, m] = end_t

                row += 1

        # Makespan = max end time across all scheduled activities
        makespan = bm.max(schedule[:, :, 4], axis=1)
        return makespan

    def decode_v2(self, seq):
        """
        Decodes a single encoded sequence and computes its completion-time matrix.

        This version performs decoding sequentially (not batched), building
        a completion-time matrix of shape (jobs, machines) by following PFSP
        constraints:
            - Jobs follow the order induced by argsort(seq).
            - Machines process jobs sequentially.
            - Each job must finish machine m before starting machine m+1.
            - Each machine must finish job j before starting job j+1.

        Parameters:
            seq(Tensor): A 1D tensor encoding a single job sequence. The order
                is determined via argsort.

        Returns:
            Tensor: The decoded sequence (after argsort), representing the job order.
        """
        self.process_time = self.data.T   # Convert to job-major form
        seq = bm.argsort(seq)

        jobs = self.process_time.shape[0]
        machines = self.process_time.shape[1]
        complete = bm.zeros((jobs, machines))

        last_number = None
        for m in range(machines):
            for j in range(seq.shape[0]):
                job_id = seq[j]

                if m == 0 and j == 0:
                    # First job on first machine
                    complete[job_id][m] = self.process_time[job_id][m]

                elif m == 0:
                    # First machine, but not first job
                    complete[job_id][m] = (
                        complete[last_number][m] + self.process_time[job_id][m]
                    )

                elif j == 0:
                    # First job on machine > 0
                    complete[job_id][m] = (
                        complete[job_id][m - 1] + self.process_time[job_id][m]
                    )

                else:
                    # General case
                    complete[job_id][m] = bm.maximum(
                        complete[job_id][m - 1],
                        complete[last_number][m]
                    ) + self.process_time[job_id][m]

                last_number = job_id

        self.time = complete
        self.ctime = complete[seq[-1]][machines - 1]  # Makespan
        return seq
    
    def visualization(self, seq):
        """
        Visualizes the schedule as a Gantt chart for the given encoded job sequence.

        This method decodes the sequence, constructs the completion-time table,
        and then generates a horizontal bar chart where each bar represents
        the processing window of a job on a machine.

        Parameters:
            seq(Tensor): A 1D tensor encoding a job sequence. It is decoded via
                `decode_v2`, and the resulting schedule is visualized.

        Returns:
            None: The method displays a matplotlib figure showing the schedule.
        """
        seq = self.decode_v2(seq)

        complete = self.time             # shape = (jobs=10, machines=30)
        process_time = self.process_time # shape = (10, 30)

        jobs = complete.shape[0]         # 10
        machines = complete.shape[1]     # 30

        # Predefined color tabl
        color1 = bm.array([
            [0,1,0],[1,0.6,0.07],[1,1,1],[0.7413,0.2845,0.8666],[0.3728,0.4740,0.0278],
            [1,0,1],[0.3488,0.2631,0.6791],[1,0.38,0],[0.7747,0.5386,0.2652],[1,1,0],
            [0.12,0.56,1],[0.99,0.9,0.79],[0.89,0.09,0.05],[0.7719,0.7270,0.7951],[0,0,1],
            [0.8020,0.4047,0.2650],[0.5,1,0],[0,1,1],[0.3754,0.7779,0.5949],[0.6158,0.6766,0.4570],
            [0.2968,0.8663,0.9846],[0.9690,0.7090,0.4807],[0.1707,0.8172,0.5249],[0.6230,0.1790,0.0942],
            [0.4387,0.4798,0.7904],[0.0003,0.2477,0.5302],[0.0873,0.7534,0.9059],[0.4257,0.3438,0.7562],
            [0.3604,0.2299,0.6952],[0.8172,0.0471,0.8199],[0.4751,0.8600,0.0291],[0.5102,0.3309,0.8741],
            [0.2829,0.7218,0.1192],[0.9618,0.1935,0.9612],[0.9291,0.7527,0.7977],[0.2128,0.7566,0.3076]
        ])

        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot Gantt bars
        for job in range(jobs):
            for m in range(machines):
                end_t = float(complete[job, m])
                duration = float(process_time[job, m])
                start_t = end_t - duration

                ax.barh(
                    m, duration, left=start_t, height=0.8,
                    color=color1[job % len(color1)],
                    edgecolor='black', linewidth=0.3
                )
                ax.text(
                    start_t + duration / 2, m, f"J{job+1}",
                    ha='center', va='center', fontsize=6
                )

        # Axes labels & formatting
        ax.set_xlabel("Work time")
        ax.set_ylabel("Work machine")
        ax.set_yticks(range(machines))
        ax.set_title(f"Finish time : {float(self.ctime)}")
        ax.invert_yaxis()

        plt.tight_layout()
        plt.show()
