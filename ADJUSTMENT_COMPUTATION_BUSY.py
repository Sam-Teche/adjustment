import pandas as pd
import numpy as np
import re
from tabulate import tabulate
import scipy.linalg as la

def input_single_value(prompt, data_type=float):
    while True:
        try:
            return data_type(input(prompt))
        except ValueError:
            print("Invalid input! Please try again.")

def get_benchmark_elevations():
    print("\nBenchmark Elevations:")
    
    # Ask for the number of benchmarks first
    while True:
        try:
            num_benchmarks = int(input("How many benchmark elevations do you need to enter? "))
            if num_benchmarks > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Initialize empty list to store benchmark elevations
    benchmarks = []
    
    # Collect each benchmark elevation
    for i in range(1, num_benchmarks + 1):
        while True:
            try:
                value = float(input(f"Enter H_B{i} elevation (m): "))
                benchmarks.append(value)
                break
            except ValueError:
                print("Please enter a valid number.")
    
    return benchmarks

def get_observation_data():
    n = input_single_value("\nNumber of observations: ", int)
    obs_numbers = []
    distances = []
    h_obs = []

    print("\nEnter Observation Numbers:")
    for i in range(n):
        obs_numbers.append(input_single_value(f"Observation #{i+1}: ", int))

    print("\nEnter Distances (Km):")
    for i in range(n):
        distances.append(input_single_value(f"Distance for Observation #{obs_numbers[i]} (Km): "))

    print("\nEnter Height Differences (h_obs):")
    for i in range(n):
        h_obs.append(input_single_value(f"Height difference for Observation #{obs_numbers[i]} (m): "))

    return pd.DataFrame({
        'Obs_No': obs_numbers,
        'Distances_Km': distances,
        'h_obs': h_obs
    })



def create_distance_matrix(df):
    n = len(df)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        distance_matrix[i, i] = df['Distances_Km'].iloc[i]
    return distance_matrix

def modify_equation(formula):
    parts = formula.replace('-', ' -').replace('+', ' +').split()
    result = []

    for part in parts:
        if part in ['+', '-']:
            result.append(part)
            continue

        operator = ''
        if part.startswith('+') or part.startswith('-'):
            operator = part[0]
            part = part[1:]
            result.append(operator)

        if re.match(r'[hH]\d+', part):
            var_match = re.match(r'([hH])(\d+)', part)
            var_index = var_match.group(2)
            result.append(f'V{var_index} + {part}')
        else:
            if '=' in part:
                left, right = part.split('=', 1)
                if re.match(r'[hH]\d+', left):
                    var_match = re.match(r'([hH])(\d+)', left)
                    var_index = var_match.group(2)
                    result.append(f'V{var_index} + {left}={right}')
                else:
                    result.append(part)
            else:
                result.append(part)

    return ' '.join(result)

def extract_coefficients(modified_eqn):
    coefficients = {}
    matches = re.finditer(r'([+-]?)\s*V(\d+)', modified_eqn)
    for match in matches:
        sign = match.group(1) or '+'
        var_index = int(match.group(2))
        coefficients[var_index] = f"{sign}1"
    return coefficients

def convert_coefficient_to_numeric(coeff_str):
    if coeff_str == '0':
        return 0
    elif coeff_str.startswith('+'):
        return 1
    elif coeff_str.startswith('-'):
        return -1
    else:
        return int(coeff_str)
    


def transpose_coefficient_matrix(coefficient_matrix):
    """
    Transposes a coefficient matrix.
    
    Parameters:
    - coefficient_matrix: A list of lists where the first column contains row labels (equation names)
      and the remaining columns contain coefficient values.
    
    Returns:
    - transposed_matrix: The transposed coefficient matrix with appropriate labels.
    """
    # Extract the equation names (row labels)
    equation_names = [row[0] for row in coefficient_matrix]
    
    # Get the number of columns in the original matrix (excluding the first column with equation names)
    num_cols = len(coefficient_matrix[0]) - 1
    
    # Create the first row of the transposed matrix with column labels
    transposed_matrix = [["Parameter"] + equation_names]
    
    # Generate the rows of the transposed matrix
    for j in range(1, num_cols + 1):
        # Create a new row starting with the parameter label (e.g., "V1", "V2", etc.)
        new_row = [f"V{j}"]
        
        # Add the coefficient values from each equation for this parameter
        for i in range(len(coefficient_matrix)):
            new_row.append(coefficient_matrix[i][j])
        
        # Add the completed row to the transposed matrix
        transposed_matrix.append(new_row)
    
    return transposed_matrix

def multiply_coefficients_with_distances(coefficient_table, distance_matrix):
    result_table = []
    for row in coefficient_table:
        equation_name = row[0]
        coefficients = row[1:]
        numeric_coeffs = [convert_coefficient_to_numeric(c) for c in coefficients]
        result_row = [equation_name]
        for i, coeff in enumerate(numeric_coeffs):
            if coeff != 0:
                weighted_coeff = coeff * distance_matrix[i, i]
                result_row.append(f"+{weighted_coeff:.3f}" if weighted_coeff > 0 else f"{weighted_coeff:.3f}")
            else:
                result_row.append("0")
        result_table.append(result_row)
    return result_table

def multiply_result_table_with_transpose(result_table):
    """
    Multiply the result_table with its transpose.
    
    Parameters:
    - result_table: A table where the first column contains equation names
                    and the remaining columns contain numeric coefficients.
    
    Returns:
    - product_table: The result of multiplying the numeric part of result_table 
                     with its transpose.
    """
    # Extract the numeric coefficients from the result_table
    numeric_matrix = []
    equation_names = []
    
    for row in result_table:
        equation_names.append(row[0])
        # Convert string coefficients to float values
        numeric_values = []
        for val in row[1:-1]:  # Exclude the last column (w values)
            try:
                # Remove '+' if present and convert to float
                numeric_values.append(float(val.replace('+', '')))
            except ValueError:
                # Handle case where value is not numeric (like '0')
                numeric_values.append(0.0)
        numeric_matrix.append(numeric_values)
    
    # Calculate transpose of numeric_matrix
    transpose = [[numeric_matrix[j][i] for j in range(len(numeric_matrix))] 
                for i in range(len(numeric_matrix[0]))]
    
    # Multiply numeric_matrix by its transpose
    product = []
    for i in range(len(numeric_matrix)):
        product_row = []
        for j in range(len(numeric_matrix)):
            # Dot product of row i from numeric_matrix and row j from transpose
            dot_product = sum(numeric_matrix[i][k] * transpose[k][j] 
                             for k in range(len(transpose)))
            product_row.append(dot_product)
        product.append(product_row)
    
    # Format the result table with equation names
    product_table = []
    for i, row in enumerate(product):
        formatted_row = [equation_names[i]]
        for val in row:
            if val == 0:
                formatted_row.append("0")
            else:
                formatted_row.append(f"+{val:.4f}" if val > 0 else f"{val:.4f}")
        product_table.append(formatted_row)
    
    return product_table

def calculate_inverse_of_product_table(product_table):
    """
    Calculate the inverse of the product table.
    
    Parameters:
    - product_table: The result of multiplying the result_table with its transpose
    
    Returns:
    - inverse_table: The inverse of the product table with equation names preserved
    """
    # Extract equation names and numeric values
    equation_names = [row[0] for row in product_table]
    
    # Create numpy array from the numeric part of the product table
    numeric_matrix = []
    for row in product_table:
        numeric_values = []
        for val in row[1:]:
            try:
                # Remove '+' if present and convert to float
                numeric_values.append(float(val.replace('+', '')))
            except ValueError:
                # Handle case where value is not numeric
                numeric_values.append(0.0)
        numeric_matrix.append(numeric_values)
    
    numeric_matrix = np.array(numeric_matrix)
    
    try:
        # Calculate the inverse using numpy
        inverse_matrix = np.linalg.inv(numeric_matrix)
        
        # Format the inverse matrix with equation names
        inverse_table = []
        for i, row in enumerate(inverse_matrix):
            formatted_row = [equation_names[i]]
            for val in row:
                if abs(val) < 1e-10:  # Treat very small values as zero
                    formatted_row.append("0")
                else:
                    formatted_row.append(f"+{val:.6f}" if val >= 0 else f"{val:.6f}")
            inverse_table.append(formatted_row)
        
        return inverse_table
    
    except np.linalg.LinAlgError as e:
        print(f"Error calculating inverse: {e}")
        print("The product table may be singular or ill-conditioned.")
        return None

def multiply_inverse_table_with_w(inverse_table, w_values):
    """
    Multiply the negative inverse of the product table with the w values.
    
    Parameters:
    - inverse_table: The inverse of the product table
    - w_values: The w values (list or array)
    
    Returns:
    - result: The result of multiplying the negative inverse_table with w_values
    """
    if inverse_table is None:
        return None
    
    # Convert w_values to numpy array if it's not already
    w_values = np.array(w_values, dtype=float)
    
    # Extract numeric values from inverse_table
    inverse_matrix = []
    for row in inverse_table:
        # Skip the first column (equation name)
        numeric_values = []
        for val in row[1:]:
            try:
                numeric_values.append(float(val.replace('+', '')))
            except ValueError:
                numeric_values.append(0.0)
        inverse_matrix.append(numeric_values)
    
    inverse_matrix = np.array(inverse_matrix)
    
    # Make the inverse matrix negative
    negative_inverse_matrix = -inverse_matrix
    
    # Multiply negative inverse matrix with w_values
    result = np.dot(negative_inverse_matrix, w_values)
    
    # Format the result
    formatted_result = []
    for i, val in enumerate(result):
        if abs(val) < 1e-10:  # Treat very small values as zero
            formatted_result.append(["K" + str(i+1), "0"])
        else:
            formatted_result.append(["K" + str(i+1), f"+{val:.8f}" if val >= 0 else f"{val:.8f}"])
    
    return formatted_result


def calculate_w_values(formulas, df, benchmarks):
    """Calculate w values by evaluating condition equations using h_obs values"""
    w_values = []
    
    for formula in formulas:
        # Fix typos in formulas (like hh4 to h4)
        formula = formula.replace('hh', 'h')
        
        # Split the formula into left and right parts
        parts = formula.split('=')
        if len(parts) != 2:
            print(f"Error: Invalid formula format: {formula}")
            continue
            
        left, right = parts
        
        left_val = 0
        right_val = 0
        
        # Process left side
        terms = re.findall(r'([+-]?)([hHbB])(\d+)', left)
        for sign, var_type, idx in terms:
            sign = sign if sign else '+'
            idx = int(idx)
            multiplier = 1 if sign == '+' else -1
            
            if var_type.lower() == 'h':
                if idx <= len(df):
                    left_val += multiplier * df['h_obs'].iloc[idx-1]
                else:
                    print(f"Error: Invalid h index {idx} in formula: {formula}")
            elif var_type.upper() == 'B':
                if idx <= len(benchmarks):
                    left_val += multiplier * benchmarks[idx-1]
                else:
                    print(f"Error: Invalid B index {idx} in formula: {formula}")
        
        # Process right side
        terms = re.findall(r'([+-]?)([hHbB])(\d+)', right)
        for sign, var_type, idx in terms:
            sign = sign if sign else '+'
            idx = int(idx)
            multiplier = 1 if sign == '+' else -1
            
            if var_type.lower() == 'h':
                if idx <= len(df):
                    right_val += multiplier * df['h_obs'].iloc[idx-1]
                else:
                    print(f"Error: Invalid h index {idx} in formula: {formula}")
            elif var_type.upper() == 'B':
                if idx <= len(benchmarks):
                    right_val += multiplier * benchmarks[idx-1]
                else:
                    print(f"Error: Invalid B index {idx} in formula: {formula}")
        
        # If the right side is a number, use that value
        if not terms and right.strip():
            try:
                right_val = eval(right.strip())
            except:
                try:
                    # Handle expressions like B2-B1
                    right = right.replace('B1', str(benchmarks[0])).replace('B2', str(benchmarks[1]))
                    right_val = eval(right)
                except:
                    print(f"Error evaluating right side: {right}")
        
        # Calculate the w value (the difference between left and right sides)
        # Multiply by -1 to match the expected convention
        w_value = -1 * (right_val - left_val)
        w_values.append(w_value)
    
    return w_values

def format_w_value(value):
    if value == 0:
        return "0"
    formatted = f"{value:.3f}".rstrip('0').rstrip('.')
    return formatted

def calculate_residual(distance_matrix, coefficient_table, formatted_result):
    """
    Calculate the residual by multiplying -P with the transpose of the coefficient table and formatted_result.
    
    Parameters:
    - P: The weight matrix (numpy array)
    - coefficient_table: The coefficient table (list of lists)
    - formatted_result: The formatted result (list of lists)
    
    Returns:
    - residual: The residual values formatted as a list of lists
    """
    # Extract numeric part of coefficient_table (excluding equation names and w column)
    numeric_matrix = []
    for row in coefficient_table:
        numeric_values = []
        for val in row[1:]:  # Exclude equation name and w column
            try:
                # Remove '+' if present and convert to float
                numeric_values.append(float(val.replace('+', '')))
            except ValueError:
                # Handle case where value is not numeric (like '0')
                numeric_values.append(0.0)
        numeric_matrix.append(numeric_values)
    
    # Convert to numpy array
    numeric_matrix = np.array(numeric_matrix)
    
    # Compute transpose of numeric_matrix
    transpose_matrix = numeric_matrix.T
    
    # Multiply -P with transpose of coefficient_table
    negative_P = -distance_matrix
    intermediate_result = np.dot(negative_P, transpose_matrix)
    
    # Extract numeric part of formatted_result and make them negative
    formatted_numeric = []
    for row in formatted_result:
        value = float(row[1].replace('+', ''))  # Remove '+' and convert to float
        formatted_numeric.append(-value)  # Make the values negative
    
    formatted_numeric = np.array(formatted_numeric)
    
    # Multiply intermediate_result with the negative formatted_result
    residual_values = np.dot(intermediate_result, formatted_numeric)
    
    # Format the residual values
    residual = []
    for i, val in enumerate(residual_values):
        if abs(val) < 1e-10:  # Treat very small values as zero
            residual.append(["Residual " + str(i+1), "0"])
        else:
            residual.append(["Residual " + str(i+1), f"+{val:.8f}" if val >= 0 else f"{val:.8f}"])
    
    return residual



def calculate_adjusted_heights(df, residual):
    
    # Extract numeric values from residuals
    residual_values = []
    for row in residual:
        try:
            # Remove '+' if present and convert to float
            residual_values.append(float(row[1].replace('+', '')))
        except ValueError:
            # Handle case where value is not numeric (like '0')
            residual_values.append(0.0)
    
    # Ensure the number of residuals matches the number of observations
    if len(residual_values) != len(df):
        print("Error: Number of residuals does not match number of observations.")
        return None
    
    # Calculate adjusted heights
    adjusted_heights = []
    for i in range(len(df)):
        h_obs = df['h_obs'].iloc[i]
        adjusted_height = h_obs + residual_values[i]
        adjusted_heights.append([f"Adjusted Height {i+1}", f"{adjusted_height:.6f}"])
    
    return adjusted_heights


def main():
    print("Condition Equation Observation Data and Calculations")
    print("="*50)
    
    # Get benchmark elevations
    benchmarks = get_benchmark_elevations()
    
    # Create a DataFrame for display
    benchmark_data = {'Point': [], 'Elevation (m)': []}
    for i, elevation in enumerate(benchmarks, 1):
        benchmark_data['Point'].append(f'B{i}')
        benchmark_data['Elevation (m)'].append(elevation)
    
    benchmark_df = pd.DataFrame(benchmark_data)
    
    # Display the table
    print("\nBenchmark Elevations:")
    print(tabulate(benchmark_df, headers='keys', tablefmt='fancy_grid', showindex=False))
    
    

    df = get_observation_data()
    df['Distances_Km'] = df['Distances_Km'].round(3)
    df['h_obs'] = df['h_obs'].round(3)
    print("\nObservation Data Table:")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))


    distance_matrix = create_distance_matrix(df)
    num_observations = int(input("\nNumber of observation: "))
    num_parameters = int(input("Number of parameter: "))
    r = num_observations - num_parameters
    print("Number of condition equation: ", r)

    formulas = []
    coefficient_table = []
    result_table = None

    if r > 0:
        print(f"\nPlease enter the {r} condition equation(s):")
        for i in range(r):
            formula = input(f"Condition equation #{i+1}: ")
            formulas.append(formula)

        print("\nModified equations:")
        for i, formula in enumerate(formulas):
            modifiedEqn = modify_equation(formula)
            print(modifiedEqn)
            coeffs = extract_coefficients(modifiedEqn)
            row = [f"Eqn {i+1}"] + [coeffs.get(j, '0') for j in range(1, num_observations + 1)]
            coefficient_table.append(row)

        headers = ["Equation"] + [f"V{i}" for i in range(1, num_observations + 1)]
        print("\nCoefficient Table:")
        print(tabulate(coefficient_table, headers, tablefmt="fancy_grid"))

        result_table = multiply_coefficients_with_distances(coefficient_table, distance_matrix)
        w_values = calculate_w_values(formulas, df, benchmarks)
        for i, row in enumerate(result_table):
            row.append(format_w_value(w_values[i]))

    print("\nDistance Matrix:")
    print(np.array2string(distance_matrix, precision=3, suppress_small=True))

    if result_table:
        headers = ["Equation"] + [f"V{i}" for i in range(1, num_observations + 1)] + ["w"]
        print("\nWeighted Coefficient Table (Coefficient × Distance):")
        print(tabulate(result_table, headers, tablefmt="fancy_grid"))

        transposed_table = transpose_coefficient_matrix(coefficient_table)
        transposed_headers = transposed_table[0]
        transposed_data = transposed_table[1:]
        print("\nTransposed Matrix:")
        print(tabulate(transposed_data, headers=transposed_headers, tablefmt="fancy_grid"))
              

        product_table = multiply_result_table_with_transpose(result_table)
        product_headers = ["Equation"] + [f"Eqn {i+1}" for i in range(len(product_table))]
        print("\nProduct Table (Result Table × Transpose):")
        print(tabulate(product_table, product_headers, tablefmt="fancy_grid"))

        inverse_table = calculate_inverse_of_product_table(product_table)
        if inverse_table:
            inverse_headers = ["Equation"] + [f"Eqn {i+1}" for i in range(len(inverse_table))]
            print("\nInverse of Product Table:")
            print(tabulate(inverse_table, inverse_headers, tablefmt="fancy_grid"))

        w_values = [float(row[-1]) for row in result_table]
    
        # Multiply inverse_table with w_values
        v_table_from_inverse = multiply_inverse_table_with_w(inverse_table, w_values)
        if v_table_from_inverse:
            print("\nK :")
            print(tabulate(v_table_from_inverse, ["Parameter", "Value"], tablefmt="fancy_grid"))

            # Calculate residual
            residual = calculate_residual( distance_matrix, coefficient_table, v_table_from_inverse)
            if residual:
                print("\nResidual :")
                print(tabulate(residual, ["Parameter", "Value"], tablefmt="fancy_grid"))


                # Calculate adjusted heights
                adjusted_heights = calculate_adjusted_heights(df, residual)
                if adjusted_heights:
                    print("\nAdjusted Heights:")
                    print(tabulate(adjusted_heights, ["Parameter", "Value"], tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()
