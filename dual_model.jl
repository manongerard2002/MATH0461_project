#=using Pkg
Pkg.add("JuMP")
Pkg.add("Gurobi")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("NamedArrays")=#
using JuMP, Gurobi, CSV, DataFrames, NamedArrays

# Data
nb_weeks = 568
capital = 500000

# Read data data from a CSV file
price_df = CSV.read("data.csv", DataFrame)
stocks_id = names(price_df)

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

# Variables
# Create a matrix of variables where p[i] <= 0 represents the shadow price for sector i
@variable(model, p[sectors_id] <= 0)
#  Create a matrix of variables where q represents the shadow price of the capital
@variable(model, q)

# Constraints
for stock in stocks_id
    @constraint(model, q + sum(p[sector] * mapping[Name(sector), stock] for sector in sectors_id) <= - mean_weekly_return[stock])
end

# Objective
@objective(model, Min, - q * capital - 0.2 * capital * sum(p[sector] for sector in sectors_id))

# Solve the model
optimize!(model)

# Print the results for each capital invested
if termination_status(model) == MOI.OPTIMAL
    println("Optimal solution found")
    obj = objective_value(model)
    println("Objective value = ", obj)
    q_value = value(q)
    println("Dual value for the capital constraint:", q_value)

    println("Dual values for each sector constraint:")
    p_values = value.(p)
    p_list = [p_values[sector] for sector in sectors_id]
    p_df = DataFrame(sector_id = sectors_id, value = p_list)
    display(p_df)
else
    println("No optimal solution found")
end
