#=using Pkg
Pkg.add("JuMP")
Pkg.add("Gurobi")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("NamedArrays")=#
using JuMP, Gurobi, CSV, DataFrames, NamedArrays

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

# Create a new JuMP model with Gurobi as the solver
model = Model(Gurobi.Optimizer)

# Variables: Create a matrix of variables where x[i] >= 0 represents the fraction of the capital invested in stock i
@variable(model, x[stocks_id] >= 0)

# Constraints
capital_constraint = @constraint(model, capital * sum(x[stock] for stock in stocks_id) == capital)

sector_constraints = Dict{Int, ConstraintRef}()
for sector in sectors_id
    sector_constraints[sector] = @constraint(model, capital * sum(x[stock] * mapping[Name(sector), stock] for stock in stocks_id) <= 0.2 * capital)
end

# Objective: Maximize the historical average weekly return
@objective(model, Max, capital * (sum(mean_weekly_return[stock] * x[stock] for stock in stocks_id) - 1))

# Solve the model
optimize!(model)

# Print the results for each capital invested
if termination_status(model) == MOI.OPTIMAL
    println("Optimal solution found")
    x_values = value.(x)
    x_list = [x_values[stock] for stock in stocks_id]
    sector_x = [sector_mapping_dict[stock] for stock in stocks_id]
    x_df = DataFrame(stock_id = stocks_id, sector = sector_x, value = x_list, capital = x_list*capital, mean_return = mean_weekly_return)
    sorted_df = sort(x_df, :mean_return, rev=true)
    df_positive = filter(row -> row[:value] > 0, sorted_df)
    println("Q2: Composition of the portfolio and means of historical return")
    display(df_positive)
    println("Objective value = ", objective_value(model), "  ", capital * sum(mean_weekly_return[stock] * x_values[stock] for stock in stocks_id))
    println(objective_value(model) - 4658.906202693179, "    ", capital * sum(mean_weekly_return[stock] * x_values[stock] for stock in stocks_id) - 504658.9062026932)

    sorted_means = sort(mean_weekly_return, rev=true)
    println("5 stocks with the highest historical return:")
    display(sorted_means[1:5])

    println("\nQ4: Optimal dual variables")
    println("Dual value for the capital constraint, q = ", shadow_price(capital_constraint))

    for sector in sort(collect(keys(sector_constraints)))
        println("Dual value for the sector $sector constraint, p_$sector = ", shadow_price(sector_constraints[sector]))
    end

    sector = 6
    println("\nQ6: Sensitivity analysis of the RHS of the limit of capital for sector $sector")
    report = lp_sensitivity_report(model)
    bounds_l6 = report[sector_constraints[sector]]
    println(report[sector_constraints[sector]])
    println("Lower bound of RHS for sector $sector in which the optimal basis stays the same = ", 0.2 - bounds_l6[1])
    println("Upper bound of RHS for sector $sector in which the optimal basis stays the same = ", 0.2 + bounds_l6[2])

    #solution_summary(model; verbose = true)
else
    println("No optimal solution found")
end
