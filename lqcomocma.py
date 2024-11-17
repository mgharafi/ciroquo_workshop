import numpy as np
from cma.logger import Logger
from cma.utilities.utils import SolutionDict
from cma.fitness_models import SurrogatePopulation, kendall_tau
from comocma import Sofomore, get_cmas
from moarchiving import BiobjectiveNondominatedSortedList
import matplotlib.pyplot as plt


class Tracer:
    """
    A solutionDict based Object, evaluates newly seen solutions and archive them or fetch and return already evaluated solutions.
    """

    def __init__(self, fun) -> None:
        self.fun = fun
        self.archive = SolutionDict()
        self.evaluations = 0

    def __call__(self, x, forget=False):
        f = self.archive.get(x, None)
        if f:
            return f
        else:
            f = np.array(self.fun(x)).tolist()
            if not forget:
                self.evaluations += 1
                self.archive[x] = f
            return f


class Surrogate_Sofomore(Sofomore):
    """
    Extends the Sofomore Framework to work with surrogate values of the uhvi
    """

    def updatCMALoggerPath(self):
        for ikernel, k in enumerate(self):
            k.logger.name_prefix = f"{self.logger.name_prefix}/cma_kernels/{ikernel}"

    def surrogate_tell(self, solutions, objective_values, surrogate_uhvi=None):
        if len(solutions) == 0:  # when asking a terminated kernel for example
            return

        if self.NDA is None:
            last_key = list(objective_values.keys())[-1]
            self.NDA = (
                BiobjectiveNondominatedSortedList
                if len(objective_values[last_key]) == 2
                else NonDominatedList
            )

        if self.reference_point is None:
            pass  # write here the max among the kernel.objective_values

        self._told_indices = []
        for ikernel, offspring in self.offspring:
            kernel = self.kernels[ikernel]
            if kernel.fit.median0 is not None and kernel.fit.median0 >= 0:
                kernel.fit.median0 = None

            kernel.tell(offspring, [surrogate_uhvi[u] for u in offspring])

            # investigate whether `kernel` hits its stopping criteria
            if kernel.stop():
                self._active_indices.remove(
                    ikernel
                )  # ikernel must be in `_active_indices`
                self._last_stopped_kernel_id = ikernel

                if self.restart is not None:
                    kernel_to_add = self.restart(self)
                    self._told_indices += [len(self)]
                    self.add(kernel_to_add)

            objective_vals = [
                objective_values.get(off)
                for off in offspring
                if not objective_values.get(off, None) is None
            ]
            kernel._last_offspring_f_values = objective_vals

        self._told_indices += [u for u, _ in self.offspring]

        current_hypervolume = self.pareto_front_cut.hypervolume
        epsilon = abs(current_hypervolume - self.best_hypervolume_pareto_front)
        if epsilon:
            self.epsilon_hypervolume_pareto_front = min(
                self.epsilon_hypervolume_pareto_front, epsilon
            )
        self.best_hypervolume_pareto_front = max(
            self.best_hypervolume_pareto_front, current_hypervolume
        )

        if self.isarchive:
            if not self.archive:
                self.archive = self.NDA(objective_vals, self.reference_point)
            else:
                self.archive.add_list(objective_vals)
        self.countiter += 1

class LQCOMO:
    def __init__(
        self,
        X0,
        sigma0,
        fitness,
        reference_point,
        inopts={},
        use_surrogate=True,
        output="exdata",
        tau_tresh=0.85,
        log=False,
        model=None,
        min_evals_percent=0,
        return_true_fitnesses=True,
        UPDATE_POLICIES={},
    ) -> None:
        # Problem specific
        self.X0 = X0
        self.num_kernels = len(X0)
        self.dimension = len(X0[0])
        self.sigma0 = sigma0
        self.reference_point = reference_point
        self.tracer = Tracer(fitness)

        # Model specific
        self.tau_tresh = tau_tresh
        self.min_evals_percent = min_evals_percent
        self.return_true_fitnesses = return_true_fitnesses
        self.UPDATE_POLICIES = UPDATE_POLICIES

        # Bookkeeping
        self.output = output
        self.real_count = 0
        self.kernels_real_count = {}
        self.log = log

        # Initialize the CMA-ES solvers
        list_of_solvers = get_cmas(
            x_starts=self.X0, sigma_starts=self.sigma0, inopts=inopts
        )

        # Initiate the surrogate-able Sofomore
        self.solver = Surrogate_Sofomore(
            list_of_solvers,
            reference_point=self.reference_point,
            opts={"verb_filename": f"{self.output}/"},
        )

        self.solver.surrogate = use_surrogate

        # Fix the path of the CMA-ES kernels loggers
        self.solver.updatCMALoggerPath()

        for ikernel, kernel in enumerate(self.solver):
            kernel.surrogate = SurrogatePopulation(self.uhvi(kernel), model=model)
            kernel.surrogate.settings.tau_truth_threshold = self.tau_tresh
            kernel.surrogate.settings.min_evals_percent = self.min_evals_percent
            kernel.surrogate.settings.return_true_fitnesses = self.return_true_fitnesses
            kernel.uhvi = self.uhvi(kernel=kernel, forget=True)

            if self.log:
                kernel.surrogate.logger = Logger(
                    kernel.surrogate,
                    path=f"{self.output}/{ikernel}",
                    labels=["ttau0", "tau0", "tau1", "evaluated_ratio", "ttau1"],
                    name="surrogate_tau",
                )

                # Overwrite the push functionality to use it outside the surrogate object
                kernel.surrogate.logger.push_, kernel.surrogate.logger.push = (
                    kernel.surrogate.logger.push,
                    lambda: None,
                )

    def uhvi(self, kernel, forget=False):
        """
        Return the mapping x --> -UHVI_i(x), which is the fitness function that CMA-ES(i) minimizes.
        """

        def wrapper(x):
            # Set the kernel to remove from the front
            self.solver.indicator_front.set_kernel(kernel, self.solver)
            # Return -uhvi(x)
            return -float(
                self.solver.indicator_front.hypervolume_improvement(
                    self.tracer(x, forget=forget)
                )
            )

        return wrapper

    def optimize(
        self,
        n_ask=1,
        user_stop=lambda: None,
        resetModel=False,
        user_updates_rules="default",
        use_incumbents=True,
        callback=None,
        injectModelOptimum=True,
    ):
        user_updates_rules = self.UPDATE_POLICIES[user_updates_rules]

        while not self.solver.stop() and not user_stop():
            solutions = self.solver.ask(n_ask)  # Ask solutions

            if self.solver.surrogate:  # Tell the surrogate case
                Fs = SolutionDict()

                # The number of told kernels (incumbent to evaluat)
                toldIndices = self.solver._told_indices[:]
                N = len(toldIndices)

                # Evaluate and update kernel's incumbent f-value and add it to archive of the surrogate if the updates rules are true
                for ikernel, incumbent in enumerate(solutions[:N]):
                    # Get the kernel object to which the incumbent belongs
                    kernel = self.solver.kernels[toldIndices[ikernel]]

                    # If the updates rules are true, evaluates the incumbent and adds it to the surrogate archive
                    if user_updates_rules(kernel):
                        # Evaluate the incumbent and update the kernel's objective value
                        kernel.objective_values = self.tracer(incumbent)

                for ikernel, incumbent in enumerate(solutions[:N]):
                    kernel = self.solver.kernels[toldIndices[ikernel]]
                    if user_updates_rules(kernel):
                        if use_incumbents:
                            incumbentValue = self.uhvi(kernel)(incumbent)

                            if incumbentValue < 0 and resetModel:
                                resetModel = False
                                if self.log:
                                    loggerT = kernel.surrogate.logger

                                kernel.surrogate = SurrogatePopulation(
                                    self.uhvi(kernel)
                                )
                                kernel.surrogate.settings.tau_truth_threshold = (
                                    self.tau_tresh
                                )
                                kernel.surrogate.settings.min_evals_percent = (
                                    self.min_evals_percent
                                )
                                kernel.surrogate.settings.return_true_fitnesses = (
                                    self.return_true_fitnesses
                                )

                                if self.log:
                                    kernel.surrogate.logger = loggerT

                            # Adds the incumbent uhvi value to the surrogate when `use_incumbent` is True
                            kernel.surrogate.model.add_data_row(
                                incumbent, incumbentValue
                            )

                for (
                    ikernel,
                    off,
                ) in self.solver.offspring:  # Evaluate the offsprings on the surrogate
                    """
                        Re-evaluate the model archive on the new UHVI
                    """
                    kernel = self.solver[ikernel]

                    if self.log:
                        tau_0 = kendall_tau(
                            [kernel.surrogate.model.eval(x) for x in off],
                            [self.uhvi(kernel=kernel, forget=True)(x) for x in off],
                        )

                        kernel.surrogate.logger.add(tau_0)

                    if user_updates_rules(kernel):
                        # Get the X from the dataset of the model
                        mX = kernel.surrogate.model.X

                        if len(mX):  # When the archive is not empty update UHVI values
                            mF = [self.uhvi(kernel=kernel)(x) for x in mX]
                            kernel.surrogate.model.F = np.hstack(mF)
                            kernel.surrogate.model.Y = np.hstack(mF)
                            kernel.surrogate.model._coefficients_count = (
                                kernel.surrogate.model.count - 1
                            )
                            coefs = kernel.surrogate.model.coefficients

                        # Evaluate offsprings on the surrogate
                        Fs_ = kernel.surrogate(off)

                        for ioff, x in enumerate(off):
                            Fs[x] = Fs_[ioff]

                        if self.log:
                            tau_1 = kendall_tau(
                                [Fs[x] for x in off],
                                [self.uhvi(kernel, forget=True)(x) for x in off],
                            )
                            kernel.surrogate.logger.add(tau_1)
                            kernel.surrogate.logger.push_()

                        self.solver.surrogate_tell(  # Update cma-es kernels in the case of Surrogate assisted optim
                            solutions=solutions,
                            objective_values=self.tracer.archive,
                            surrogate_uhvi=Fs,
                        )

                        for k in self.solver._told_indices:
                            kernel = self.solver[k]
                            kernel.countiter -= 1
                            # Get the current kernel's number of true function evaluations
                            if injectModelOptimum:
                                kernel.inject([kernel.surrogate.model.xopt])

                            true_evaluations = kernel.surrogate.evals.evaluations

                            # Increment the count of the kernel's true funciton evaluations
                            if k in self.kernels_real_count:
                                self.kernels_real_count[k] += true_evaluations
                            else:
                                self.kernels_real_count[k] = true_evaluations

                            # Update the kernel's number of true function evaluations
                            kernel.countiter += 1
                            kernel.countevals = self.kernels_real_count[k]
                            kernel.logger.add()

                        # Update the overall count of function evaluations
                        self.solver.countevals = self.tracer.evaluations

                    else:
                        # Evaluates the solutions on the model without updating the model
                        for x in off:
                            Fs[x] = kernel.surrogate.model.eval(x)

                        self.solver.surrogate_tell(  # Update cma-es kernels in the case of Surrogate assisted optim
                            solutions=solutions,
                            objective_values=self.tracer.archive,
                            surrogate_uhvi=Fs,
                        )

                        true_evaluations = 0

                        # Increment the count of the kernel's true funciton evaluations
                        if k in self.kernels_real_count:
                            self.kernels_real_count[k] += true_evaluations
                        else:
                            self.kernels_real_count[k] = true_evaluations

                        kernel.countevals = self.kernels_real_count[k]
                        kernel.logger.add()

                        self.solver.countevals = self.tracer.evaluations

                        if self.log:
                            tau_1 = kendall_tau(
                                [Fs[x] for x in off],
                                [self.uhvi(kernel, forget=True)(x) for x in off],
                            )
                            kernel.surrogate.logger.add(-2)
                            kernel.surrogate.logger.add(-2)
                            kernel.surrogate.logger.add(0)
                            kernel.surrogate.logger.add(tau_1)
                            kernel.surrogate.logger.push_()

            else:  # Tell without surrogate surrogate
                F = [self.tracer(x) for x in solutions]
                self.solver.tell(  # Update cma-es kernels in the case of true function evaluation
                    solutions=solutions, objective_values=F
                )

            if self.log:
                self.solver.logger.add()
                self.solver.disp()

            if callback:
                callback()

    def plotSurrogate(
        self, kernel, ax, drawCurves=["ttau0", "tau0", "tau1", "neval", "ttau1"]
    ):
        para = ax.twinx()
        lw = 0.8
        surrogate = kernel.surrogate
        surrogate.logger.load()
        ttau0, tau0, tau1, neval, ttau1 = surrogate.logger.data.T
        data = [ttau0, tau0, tau1, neval, ttau1]
        labels = ["ttau0", "tau0", "tau1", "neval", "ttau1"]
        axs = [ax, ax, ax, para, ax]
        styles = [
            {"lw": lw},
            {"ls": "--", "lw": lw},
            {"ls": "--", "lw": lw},
            {"c": "red", "lw": lw},
            {"lw": lw},
        ]
        curves = dict(zip(labels, zip(data, axs, styles)))

        X = range(len(ttau0))

        for L in drawCurves:
            Y, A, S = curves[L]
            A.plot(X, Y, **S, label=L)

        ax.axhline(
            y=surrogate.settings.tau_truth_threshold,
            ls=":",
            alpha=0.4,
            c="black",
            lw=lw,
        )

        # para.plot(X, neval, label='nevals ratio', lw=lw)
        ax.set_ylim(top=1, bottom=-1)
        ax.set_xlabel("iterations")
        para.set_ylim(top=2, bottom=0)
        ax.grid(True)
        ax.legend()
        para.legend()

    def plotGapUHVI(self, kernel, ax):
        log = kernel.logger
        log.load()
        xs = log.data["xmean"][:, 5:]
        ff = self.uhvi(kernel, forget=True)

        X = range(len(xs))
        Y = [-ff(x) for x in xs]
        Ymax = max(Y)
        Y = np.minimum.accumulate([-(y - Ymax) for y in Y])

        ax.plot(X, Y)
        ax.minorticks_on()
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="both")
        # ax.set_ylim()
        ax.set_xlabel("iterations")
        ax.set_title(f"Gap to the best UHVI cHV(opt) = {float(Ymax):e}")

    def plot_kernel(self, kernel):
        kernel.plot(
            addcols=1,
            xsemilog=True,
            iabscissa=0,
        )

        plt.subplot(2, 3, 6)
        ax = plt.gca()
        ax.clear()
        self.plotSurrogate(kernel, ax=ax)

        plt.subplot(2, 3, 3)
        ax = plt.gca()
        ax.clear()
        self.plotGapUHVI(kernel, ax=ax)
