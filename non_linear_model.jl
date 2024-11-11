#=using Pkg
Pkg.add("JuMP")
Pkg.add("Gurobi")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("NamedArrays")
Pkg.add("MathOptInterface")
Pkg.add("Plots")
Pkg.add("LaTeXStrings")
Pkg.add("Statistics")=#
using JuMP, Gurobi, CSV, DataFrames, NamedArrays, MathOptInterface, Plots, LaTeXStrings, Statistics

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

#correct ?
#covariance_matrix = cov(mean_weekly_return)
#display(covariance_matrix)

covariance_matrix = NamedArray(cov(Matrix(weekly_return)'), (stocks_id, stocks_id), ("Stocks", "Stocks"))
display(covariance_matrix)

gamma_df = CSV.read("gamma_vals.csv", DataFrame; header=false)

efficient_frontier = DataFrame(gamma=Float64[], expected_return=Float64[], risk=Float64[])

for gamma in gamma_df[:, 1]
    # Create a new JuMP model with Gurobi as the solver
    local model = Model(Gurobi.Optimizer)

    # Variables: Create a matrix of variables where x[i] in [0, 1] represents the fraction of the capital invested in stock i
    @variable(model, 0 <= x[stocks_id] <= 1)

    # Constraints
    local capital_constraint = @constraint(model, capital * sum(x[stock] for stock in stocks_id) <= capital)

    local sector_constraints = Dict{Int, ConstraintRef}()
    for sector in sectors_id
        sector_constraints[sector] = @constraint(model, capital * sum(x[stock] * mapping[Name(sector), stock] for stock in stocks_id) <= 0.2 * capital)
    end

    # Objective: Maximize the utility, representing the trade-off between the average historical return and the risk of the portfolio
    @objective(model, Max, capital * (sum(mean_weekly_return[stock] * x[stock] for stock in stocks_id) - 1) - gamma * sum(x[i] * covariance_matrix[i, j] * x[j] for i in stocks_id, j in stocks_id))
    #x' * covariance_matrix * x same as sum(x[i] * covariance_matrix[i, j] * x[j] for i in stocks_id, j in stocks_id)

    # Solve the model
    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        println("Optimal solution found")
        local x_values = value.(x)
        local x_list = [x_values[stock] for stock in stocks_id]
        local sector_x = [sector_mapping_dict[stock] for stock in stocks_id]
        local x_df = DataFrame(stock_id = stocks_id, sector = sector_x, value = x_list*100, capital = x_list*capital, mean_return = mean_weekly_return)
        local sorted_df = sort(x_df, :mean_return, rev=true)
        local df_positive = filter(row -> row[:value] > 0, sorted_df)
        println("Q2: Composition of the portfolio and means of historical return")
        display(df_positive)
        local objective = objective_value(model)
        println("Objective value = ", objective)

        # Store results for plotting the efficient frontier
        expected_return = capital * (sum(mean_weekly_return[stock] * x_values[stock] for stock in stocks_id) - 1)
        risk = Vector(x_values)' * Matrix(covariance_matrix) * Vector(x_values)
        #println(Vector(x_values)' * Matrix(covariance_matrix) * Vector(x_values))
        #println(sum(x_values[i] * covariance_matrix[i, j] * x_values[j] for i in stocks_id, j in stocks_id))
        push!(efficient_frontier, (gamma=gamma, expected_return=expected_return, risk=risk))
    else
        println("No optimal solution found for gamma =", gamma)
    end
end

display(efficient_frontier)

plot(efficient_frontier.risk, efficient_frontier.expected_return, xlabel="Risk (Portfolio Variance)", ylabel="Expected Return", title="Efficient Frontier", legend=false)
plt2 = plot!(twiny(), efficient_frontier.gamma, efficient_frontier.expected_return, xlabel=L"$\gamma$", legend = false)
savefig(plt2, "Q8_plot.pdf")
