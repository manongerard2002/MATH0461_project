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

#=mapping = Dict{Int, Vector{String}}() # access with mapping[3] where 3 is a number of a stock
for row in eachrow(sector_mapping_df)
    stock = row[1]
    sector = row[2]

    if haskey(mapping, sector)
        push!(mapping[sector], stock)
    else
        mapping[sector] = [stock]
    end
end=#

weekly_return = NamedArray(zeros(Float64, length(stocks_id), nb_weeks - 1), (stocks_id, 2:nb_weeks), ("Stocks", "Weeks"))
for stock in stocks_id, week in 2:nb_weeks
    weekly_return[stock, Name(week)] = (price_df[week, stock] - price_df[week-1, stock]) / price_df[week-1, stock] * 100
end

mean_weekly_return = NamedArray(zeros(Float64, length(stocks_id)), (stocks_id), ("Stocks"))
for stock in stocks_id
    mean_weekly_return[stock] = sum(weekly_return[stock, Name(week)] for week in 2:nb_weeks)
end

# Create a new JuMP model with Gurobi as the solver
model = Model(Gurobi.Optimizer)

# Variables: Create a matrix of variables where x[i] in [0, capital] represents thee capital invested in stock i
@variable(model, 0 <= x[stocks_id] <= capital)

# Constraints
@constraint(model, sum(x[stock] for stock in stocks_id) <= capital)

for sector in sectors_id
    #@constraint(model, sum(x[stock] for stock in mapping[sector]) <= 0.2 * capital)
    @constraint(model, sum(x[stock] * mapping[Name(sector), stock] for stock in stocks_id) <= 0.2 * capital)
end

# Objective: Maximize the historical average weekly return
@objective(model, Max, sum(mean_weekly_return[stock] * x[stock] for stock in stocks_id))

# Solve the model
optimize!(model)

# Print the results for each capital invested
if termination_status(model) == MOI.OPTIMAL
    println("Optimal solution found")
    x_values = value.(x)
    x_list = [x_values[stock] for stock in stocks_id]
    sector_x = [sector_mapping_dict[stock] for stock in stocks_id]
    x_df = DataFrame(stock_id = stocks_id, sector = sector_x, value = x_list, mean_return = mean_weekly_return)
    sorted_df = sort(x_df, :mean_return, rev=true)
    df_positive = filter(row -> row[:value] > 0, sorted_df)
    display(df_positive)

    sorted_means = sort(mean_weekly_return, rev=true)
    display(sorted_means[1:5])
else
    println("No optimal solution found")
end
