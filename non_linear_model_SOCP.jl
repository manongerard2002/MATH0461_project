#=using Pkg
Pkg.add("JuMP")
Pkg.add("Gurobi")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("NamedArrays")
Pkg.add("Plots")
Pkg.add("Statistics")
Pkg.add("LinearAlgebra")=#
using JuMP, Gurobi, CSV, DataFrames, NamedArrays, Plots, Statistics, LinearAlgebra

# Data
capital = 500000

# Read data data from a CSV file
price_df = CSV.read("data.csv", DataFrame)
stocks_id = names(price_df)
nb_weeks = length(price_df[:, 1])

sector_mapping_df = CSV.read("sector_mapping.csv", DataFrame; header=false, types=[String, Int])
sector_mapping_dict = Dict(row.Column1 => row.Column2 for row in eachrow(sector_mapping_df))

sectors_id = sort(unique(sector_mapping_df[:, 2])) # Note: starts at 0

mapping = NamedArray(zeros(Int, length(sectors_id), length(stocks_id)), (sectors_id, stocks_id), ("Sectors", "Stocks"))
for row in eachrow(sector_mapping_df)
    mapping[Name(row[2]), row[1]] = 1
end

weekly_return = NamedArray(zeros(Float64, length(stocks_id), nb_weeks - 1), (stocks_id, 2:nb_weeks), ("Stocks", "Weeks"))
for stock in stocks_id, week in 2:nb_weeks
    weekly_return[stock, Name(week)] = (price_df[week, stock] - price_df[week-1, stock]) / price_df[week-1, stock] * 100
end

mean_weekly_return = NamedArray(zeros(Float64, length(stocks_id)), (stocks_id), ("Stocks"))
for stock in stocks_id
    mean_weekly_return[stock] = sum(weekly_return[stock, Name(week)] for week in 2:nb_weeks) / (nb_weeks - 1)
end

Q = NamedArray(cov(Matrix(weekly_return)'), (stocks_id, stocks_id), ("Stocks", "Stocks"))
R = cholesky(Q).U

gamma_df = CSV.read("gamma_vals.csv", DataFrame; header=false)

efficient_frontier = DataFrame(gamma=Float64[], expected_return=Float64[], risk=Float64[])

# Create a new JuMP model with Gurobi as the solver
model = Model(Gurobi.Optimizer)
set_optimizer_attribute(model, "OutputFlag", 0)

# Variables: Create a matrix of variables where x[i] in [0, 1] represents the fraction of the capital invested in stock i
@variable(model, 0 <= x[stocks_id] <= 1)

# Auxiliary variable for SOCP
@variable(model, t)
@variable(model, Rx[1:size(R, 1)])

# Add variables expected_return to be able to retrieve its value
@variable(model, expected_return)
@constraint(model, expected_return == sum(mean_weekly_return[stock] * x[stock] for stock in stocks_id))

# Constraints
@constraint(model, sum(x[stock] for stock in stocks_id) <= 1)

for sector in sectors_id
    @constraint(model, sum(x[stock] * mapping[Name(sector), stock] for stock in stocks_id) <= 0.2)
end

for i in 1:size(R, 1)
    @constraint(model, Rx[i] == sum(R[i, j] * x[stock] for (j, stock) in enumerate(stocks_id)))
end

@constraint(model, [(t + 1)/2; Rx; (t - 1)/2] in SecondOrderCone())

for gamma in gamma_df[:, 1]
    # Objective: Maximize the utility, representing the trade-off between the average historical return and the risk of the portfolio
    @objective(model, Max, expected_return - gamma * t)

    # Solve the model
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        println("Optimal solution found for γ = ", gamma)

        # Store results for plotting the efficient frontier
        Rx_values = value.(Rx)
        push!(efficient_frontier, (gamma=gamma, expected_return=value(expected_return), risk=Rx_values'Rx_values))
    else
        println("No optimal solution found for γ = ", gamma)
    end
end
display(efficient_frontier)

plt = scatter(efficient_frontier.risk, efficient_frontier.expected_return, xlabel="Risk (Portfolio Variance)", ylabel="Expected Return [%]", label="", legend=:bottomright)
plot(plt, efficient_frontier.risk, efficient_frontier.expected_return, color=:blue, label="efficient frontier")

s = size(efficient_frontier, 1)
annotate!(efficient_frontier.risk[1] - 1.2, efficient_frontier.expected_return[1] - 0.03, text("γ=" * string(efficient_frontier.gamma[1]), :left, 8))
for i in 3:2:8
    annotate!(efficient_frontier.risk[i], efficient_frontier.expected_return[i] - 0.03, text("γ=" * string(efficient_frontier.gamma[i]), :left, 8))
end
for i in 2:2:8
    annotate!(efficient_frontier.risk[i] - 0.4, efficient_frontier.expected_return[i] + 0.02, text("γ=" * string(efficient_frontier.gamma[i]), :right, 8))
end
for i in 9:s-2
    annotate!(efficient_frontier.risk[i] - 0.4, efficient_frontier.expected_return[i], text("γ=" * string(efficient_frontier.gamma[i]), :right, 8))
end
for i in s-1:s
    annotate!(efficient_frontier.risk[i] + 0.4, efficient_frontier.expected_return[i], text("γ=" * string(efficient_frontier.gamma[i]), :left, 8))
end
savefig("Q8_SOCP_plot.pdf")