#=using Pkg
Pkg.add("JuMP")
Pkg.add("Gurobi")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("NamedArrays")
Pkg.add("MathOptInterface")
Pkg.add("Plots")
Pkg.add("LaTeXStrings")=#
using JuMP, Gurobi, CSV, DataFrames, NamedArrays, MathOptInterface, Plots, LaTeXStrings

# Data
capital = 500000

# Read data data from a CSV file
price_df = CSV.read("data.csv", DataFrame)
stocks_id = names(price_df)
nb_weeks = length(price_df[:, 1])

sector_mapping_df = CSV.read("sector_mapping.csv", DataFrame; header=false, types=[String, Int])
sector_mapping_dict = Dict(row.Column1 => row.Column2 for row in eachrow(sector_mapping_df))

sectors_id = sort(unique(sector_mapping_df[:, 2]))

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

#method = 0 for primal simplex; 1 for dual simplex
#presolve = -1 for default; 0 to deactivate
#output = 1 for default; 0 to deactivate
function solveModel(method, presolve=-1, output=0)
    # Create a new JuMP model with Gurobi as the solver
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", output)
    set_optimizer_attribute(model, "Method", method)
    set_optimizer_attribute(model, "Presolve", presolve)

    # Variables: Create a matrix of variables where x[i] in [0, 1] represents the fraction of the capital invested in stock i
    @variable(model, 0 <= x[stocks_id] <= 1)

    # Constraints
    capital_constraint = @constraint(model, sum(x[stock] for stock in stocks_id) <= 1)

    sector_constraints = Dict{Int, ConstraintRef}()
    for sector in sectors_id
        sector_constraints[sector] = @constraint(model, sum(x[stock] * mapping[Name(sector), stock] for stock in stocks_id) <= 0.2)
    end

    # Objective: Maximize the historical average weekly return
    @objective(model, Max, sum(mean_weekly_return[stock] * x[stock] for stock in stocks_id))

    # Solve the model
    optimize!(model)

    return model, capital_constraint, sector_constraints, x
end

model, capital_constraint, sector_constraints, x = solveModel(0)
# Print the results for each capital invested
if termination_status(model) == MOI.OPTIMAL
    println("Optimal solution found")
    println("------------------------------------------------------------------------")
    println("Q2: Composition of the portfolio and means of historical return")
    println("------------------------------------------------------------------------")
    x_values = value.(x)
    x_list = [x_values[stock] for stock in stocks_id]
    sector_x = [sector_mapping_dict[stock] for stock in stocks_id]
    x_df = DataFrame(stock_id = stocks_id, sector = sector_x, value = x_list*100, capital = x_list*capital, mean_weekly_return = mean_weekly_return)
    sorted_df = sort(x_df, :mean_weekly_return, rev=true)
    df_positive = filter(row -> row[:value] > 0, sorted_df)
    display(df_positive)
    objective = objective_value(model)
    println("Objective value = ", objective)

    v_stat = Dict(
           xi => MOI.get(model, MOI.VariableBasisStatus(), xi)
           for xi in all_variables(model)
       )
    basis = []
    for stock in stocks_id
        if v_stat[x[stock]] == MathOptInterface.BASIC
            push!(basis, stock)
        end
    end
    println("Basis = ", basis)

    sorted_means = sort(mean_weekly_return, rev=true)
    println("5 stocks with the highest historical return:")
    display(sorted_means[1:5])

    println("\n\n------------------------------------------------------------------------")
    println("Q4: Optimal dual variables")
    println("------------------------------------------------------------------------")
    sector_sorted = sort(collect(keys(sector_constraints)))
    q = shadow_price(capital_constraint)
    println("Dual value for the capital constraint, q = ", q)
    p = NamedArray(zeros(Float64, length(sectors_id)), (sectors_id), ("Sectors"))
    for sector in sector_sorted
        p[Name(sector)] = shadow_price(sector_constraints[sector])
        println("Dual value for the sector $sector constraint, p_$sector = ", p[Name(sector)])
    end

    # Uncomment this part to see the values of the duals that here verified
    #=q_computed = 0
    p_computed = NamedArray(zeros(Float64, length(sectors_id)), (sectors_id), ("Sectors"))
    delta_increase = 0.1
    set_normalized_rhs(capital_constraint, normalized_rhs(capital_constraint) + delta_increase)
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        q_computed = (objective_value(model) - objective) / delta_increase
    end
    set_normalized_rhs(capital_constraint, normalized_rhs(capital_constraint) - delta_increase)
    optimize!(model)

    p_computed = NamedArray(zeros(Float64, length(sectors_id)), (sectors_id), ("Sectors"))
    delta_decrease = -0.01
    for sector in sector_sorted
        local constraint = sector_constraints[sector]
        set_normalized_rhs(constraint, normalized_rhs(constraint) + delta_decrease)
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            p_computed[Name(sector)] = (objective_value(model) - objective) / delta_decrease
        end
        set_normalized_rhs(constraint, normalized_rhs(constraint) - delta_decrease)
    end
    optimize!(model)

    sector_names = vcat("capital constraint", ["sector $sector constraint" for sector in sector_sorted])
    p_values = vcat([q], [p[Name(sector)] for sector in sector_sorted])
    p_computed_values = vcat([q_computed], [p_computed[Name(sector)] for sector in sector_sorted])
    dual_df = DataFrame(
        sector = sector_names,
        p = p_values,
        p_computed = p_computed_values,
        difference = p_values .- p_computed_values
    )
    display(dual_df)=#

    println("\n\n------------------------------------------------------------------------")
    println("Q5: Primal vs Dual simplex algorithms")
    println("------------------------------------------------------------------------")
    primal_model, _, _, _ = solveModel(0, 0, 1)
    println("Primal simplex number of iterations = ", simplex_iterations(primal_model))
    println("------------------------------------------------------------------------")
    dual_model, _, _, _ = solveModel(1, 0, 1)
    println("Dual simplex number of iterations = ", simplex_iterations(dual_model))

    println("\n\n------------------------------------------------------------------------")
    sector = 6
    println("Q6: Sensitivity analysis of the RHS of the limit of capital for sector $sector")
    println("------------------------------------------------------------------------")
    report = lp_sensitivity_report(model)
    constraint = sector_constraints[sector]
    decrease, increase = report[constraint]
    println("The optimal basis stays the same for:")
    println("∆l_6 = [", decrease, ",", increase, "]")
    rhs_6 = normalized_rhs(sector_constraints[sector])
    println("l_6 = [", rhs_6 + decrease, ",", rhs_6 + increase, "]")

    println("New solution for when the optimal basis stays the same ranges from ", objective + p[Name(sector)] * decrease, " to ", objective + p[Name(sector)] * increase)

    x_axis = decrease:0.1:increase
    y_axis = objective .+ p[Name(sector)] * x_axis
    x_axis_2 = (decrease + rhs_6):0.1:(increase + rhs_6)

    plot(x_axis, y_axis, xlabel=L"$\Delta l_6$", ylabel="Expected Return [%]", label=L"$1.009318 + 0.40013 \times \Delta l_6$")
    plt2 = plot!(twiny(), x_axis_2, y_axis, xlabel=L"$l_6$", legend = false)
    savefig(plt2, "Q6_plot.pdf")
    
    # Uncomment to optimize the problem using the lower bound of delta l_6 in order to check our result 
    #=set_normalized_rhs(constraint, normalized_rhs(constraint) + decrease)
    optimize!(model)
    println("lower bound = ", objective_value(model))
    x_values = value.(x)
    x_list = [x_values[stock] for stock in stocks_id]
    x_df = DataFrame(stock_id = stocks_id, sector = sector_x, value = x_list*100, capital = x_list*capital, mean_return = mean_weekly_return)
    sorted_df = sort(x_df, :mean_return, rev=true)
    df_positive = filter(row -> row[:value] > 0, sorted_df)
    display(df_positive)

    v_stat = Dict(
           xi => MOI.get(model, MOI.VariableBasisStatus(), xi)
           for xi in all_variables(model)
       )
    new_basis = []
    for stock in stocks_id
        if v_stat[x[stock]] == MathOptInterface.BASIC
            push!(new_basis, stock)
        end
    end
    println("Basis = ", new_basis)
    #set_normalized_rhs(constraint, normalized_rhs(constraint) - decrease)=#
else
    println("No optimal solution found")
end
