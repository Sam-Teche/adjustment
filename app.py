from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import scipy.linalg as la
from flask_cors import CORS

from flask import render_template

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Helper functions from original script
def create_distance_matrix(df):
    """Create a distance matrix from the dataframe"""
    n = len(df)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        distance_matrix[i, i] = df['Distances_Km'].iloc[i]
    return distance_matrix

def modify_equation(formula):
    """Modify equation to standard format"""
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
    """Extract coefficients from the modified equation"""
    coefficients = {}
    matches = re.finditer(r'([+-]?)\s*V(\d+)', modified_eqn)
    for match in matches:
        sign = match.group(1) or '+'
        var_index = int(match.group(2))
        coefficients[var_index] = f"{sign}1"
    return coefficients

def convert_coefficient_to_numeric(coeff_str):
    """Convert coefficient string to numeric value"""
    if coeff_str == '0':
        return 0
    elif coeff_str.startswith('+'):
        return 1
    elif coeff_str.startswith('-'):
        return -1
    else:
        return int(coeff_str)

def transpose_coefficient_matrix(coefficient_matrix):
    """Transpose a coefficient matrix"""
    equation_names = [row[0] for row in coefficient_matrix]
    num_cols = len(coefficient_matrix[0]) - 1
    transposed_matrix = [["Parameter"] + equation_names]
    
    for j in range(1, num_cols + 1):
        new_row = [f"V{j}"]
        for i in range(len(coefficient_matrix)):
            new_row.append(coefficient_matrix[i][j])
        transposed_matrix.append(new_row)
    
    return transposed_matrix

def multiply_coefficients_with_distances(coefficient_table, distance_matrix):
    """Multiply coefficients with distances"""
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
    """Multiply the result table with its transpose"""
    numeric_matrix = []
    equation_names = []
    
    for row in result_table:
        equation_names.append(row[0])
        numeric_values = []
        for val in row[1:-1]:  # Exclude the last column (w values)
            try:
                numeric_values.append(float(val.replace('+', '')))
            except ValueError:
                numeric_values.append(0.0)
        numeric_matrix.append(numeric_values)
    
    transpose = [[numeric_matrix[j][i] for j in range(len(numeric_matrix))] 
                for i in range(len(numeric_matrix[0]))]
    
    product = []
    for i in range(len(numeric_matrix)):
        product_row = []
        for j in range(len(numeric_matrix)):
            dot_product = sum(numeric_matrix[i][k] * transpose[k][j] 
                             for k in range(len(transpose)))
            product_row.append(dot_product)
        product.append(product_row)
    
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
    """Calculate the inverse of the product table"""
    equation_names = [row[0] for row in product_table]
    numeric_matrix = []
    
    for row in product_table:
        numeric_values = []
        for val in row[1:]:
            try:
                numeric_values.append(float(val.replace('+', '')))
            except ValueError:
                numeric_values.append(0.0)
        numeric_matrix.append(numeric_values)
    
    numeric_matrix = np.array(numeric_matrix)
    
    try:
        inverse_matrix = np.linalg.inv(numeric_matrix)
        
        inverse_table = []
        for i, row in enumerate(inverse_matrix):
            formatted_row = [equation_names[i]]
            for val in row:
                if abs(val) < 1e-10:
                    formatted_row.append("0")
                else:
                    formatted_row.append(f"+{val:.6f}" if val >= 0 else f"{val:.6f}")
            inverse_table.append(formatted_row)
        
        return inverse_table
    
    except np.linalg.LinAlgError as e:
        return None

def multiply_inverse_table_with_w(inverse_table, w_values):
    """Multiply the negative inverse of the product table with the w values"""
    if inverse_table is None:
        return None
    
    w_values = np.array(w_values, dtype=float)
    inverse_matrix = []
    
    for row in inverse_table:
        numeric_values = []
        for val in row[1:]:
            try:
                numeric_values.append(float(val.replace('+', '')))
            except ValueError:
                numeric_values.append(0.0)
        inverse_matrix.append(numeric_values)
    
    inverse_matrix = np.array(inverse_matrix)
    negative_inverse_matrix = -inverse_matrix
    result = np.dot(negative_inverse_matrix, w_values)
    
    formatted_result = []
    for i, val in enumerate(result):
        if abs(val) < 1e-10:
            formatted_result.append(["K" + str(i+1), "0"])
        else:
            formatted_result.append(["K" + str(i+1), f"+{val:.8f}" if val >= 0 else f"{val:.8f}"])
    
    return formatted_result

def calculate_w_values(formulas, df, benchmarks):
    """Calculate w values by evaluating condition equations using h_obs values"""
    w_values = []
    
    for formula in formulas:
        formula = formula.replace('hh', 'h')
        parts = formula.split('=')
        
        if len(parts) != 2:
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
            elif var_type.upper() == 'B':
                if idx <= len(benchmarks):
                    left_val += multiplier * benchmarks[idx-1]
        
        # Process right side
        terms = re.findall(r'([+-]?)([hHbB])(\d+)', right)
        for sign, var_type, idx in terms:
            sign = sign if sign else '+'
            idx = int(idx)
            multiplier = 1 if sign == '+' else -1
            
            if var_type.lower() == 'h':
                if idx <= len(df):
                    right_val += multiplier * df['h_obs'].iloc[idx-1]
            elif var_type.upper() == 'B':
                if idx <= len(benchmarks):
                    right_val += multiplier * benchmarks[idx-1]
        
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
                    pass
        
        # Calculate the w value
        w_value = -1 * (right_val - left_val)
        w_values.append(w_value)
    
    return w_values

def format_w_value(value):
    """Format w value as string"""
    if value == 0:
        return "0"
    formatted = f"{value:.3f}".rstrip('0').rstrip('.')
    return formatted

def calculate_residual(distance_matrix, coefficient_table, formatted_result):
    """Calculate the residual values"""
    numeric_matrix = []
    
    for row in coefficient_table:
        numeric_values = []
        for val in row[1:]:
            try:
                numeric_values.append(float(val.replace('+', '')))
            except ValueError:
                numeric_values.append(0.0)
        numeric_matrix.append(numeric_values)
    
    numeric_matrix = np.array(numeric_matrix)
    transpose_matrix = numeric_matrix.T
    negative_P = -distance_matrix
    intermediate_result = np.dot(negative_P, transpose_matrix)
    
    formatted_numeric = []
    for row in formatted_result:
        value = float(row[1].replace('+', ''))
        formatted_numeric.append(-value)
    
    formatted_numeric = np.array(formatted_numeric)
    residual_values = np.dot(intermediate_result, formatted_numeric)
    
    residual = []
    for i, val in enumerate(residual_values):
        if abs(val) < 1e-10:
            residual.append(["Residual " + str(i+1), "0"])
        else:
            residual.append(["Residual " + str(i+1), f"+{val:.8f}" if val >= 0 else f"{val:.8f}"])
    
    return residual

def calculate_adjusted_heights(df, residual):
    """Calculate adjusted heights"""
    residual_values = []
    
    for row in residual:
        try:
            residual_values.append(float(row[1].replace('+', '')))
        except ValueError:
            residual_values.append(0.0)
    
    if len(residual_values) != len(df):
        return None
    
    adjusted_heights = []
    for i in range(len(df)):
        h_obs = df['h_obs'].iloc[i]
        adjusted_height = h_obs + residual_values[i]
        adjusted_heights.append([f"Adjusted Height {i+1}", f"{adjusted_height:.6f}"])
    
    return adjusted_heights


@app.route('/')
def index():
    return render_template('index.html')



# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "API is running"}), 200

@app.route('/api/adjustment', methods=['POST'])
def perform_adjustment():
    """Main endpoint for adjustment computation"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['benchmarks', 'observations', 'num_parameters', 'formulas']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Process input data
        benchmarks = data['benchmarks']
        observations = data['observations']
        num_parameters = data['num_parameters']
        formulas = data['formulas']
        
        # Create DataFrame from observations
        df = pd.DataFrame(observations)
        
        # Validate DataFrame columns
        required_columns = ['Obs_No', 'Distances_Km', 'h_obs']
        for col in required_columns:
            if col not in df.columns:
                return jsonify({"error": f"Missing column in observations: {col}"}), 400
        
        # Ensure data types
        df['Distances_Km'] = df['Distances_Km'].astype(float)
        df['h_obs'] = df['h_obs'].astype(float)
        
        # Calculate the number of observations and condition equations
        num_observations = len(df)
        r = num_observations - num_parameters
        
        # Return error if no condition equations needed but formulas provided
        if r <= 0 and formulas:
            return jsonify({"error": "No condition equations needed but formulas provided"}), 400
        
        # Return error if condition equations needed but no formulas provided
        if r > 0 and (not formulas or len(formulas) != r):
            return jsonify({"error": f"Expected {r} condition equations but got {len(formulas) if formulas else 0}"}), 400
        
        # Perform calculations
        distance_matrix = create_distance_matrix(df)
        
        # Initialize results dictionary
        results = {
            "num_observations": num_observations,
            "num_parameters": num_parameters,
            "num_condition_equations": r,
            "distance_matrix": distance_matrix.tolist(),
            "observation_data": df.to_dict(orient='records'),
            "benchmark_data": [{"Point": f"B{i+1}", "Elevation": b} for i, b in enumerate(benchmarks)]
        }
        
        if r > 0:
            # Process formulas and create coefficient table
            coefficient_table = []
            modified_equations = []
            
            for i, formula in enumerate(formulas):
                modifiedEqn = modify_equation(formula)
                modified_equations.append(modifiedEqn)
                coeffs = extract_coefficients(modifiedEqn)
                row = [f"Eqn {i+1}"] + [coeffs.get(j, '0') for j in range(1, num_observations + 1)]
                coefficient_table.append(row)
            
            # Calculate weighted coefficient table
            result_table = multiply_coefficients_with_distances(coefficient_table, distance_matrix)
            w_values = calculate_w_values(formulas, df, benchmarks)
            
            for i, row in enumerate(result_table):
                row.append(format_w_value(w_values[i]))
            
            # Calculate transposed matrix
            transposed_table = transpose_coefficient_matrix(coefficient_table)
            
            # Calculate product table
            product_table = multiply_result_table_with_transpose(result_table)
            
            # Calculate inverse of product table
            inverse_table = calculate_inverse_of_product_table(product_table)
            
            # Calculate K values
            v_table_from_inverse = None
            residual = None
            adjusted_heights = None
            
            if inverse_table:
                w_values = [float(row[-1]) for row in result_table]
                v_table_from_inverse = multiply_inverse_table_with_w(inverse_table, w_values)
                
                if v_table_from_inverse:
                    residual = calculate_residual(distance_matrix, coefficient_table, v_table_from_inverse)
                    
                    if residual:
                        adjusted_heights = calculate_adjusted_heights(df, residual)
            
            # Add all calculation results to the results dictionary
            results.update({
                "modified_equations": modified_equations,
                "coefficient_table": coefficient_table,
                "result_table": result_table,
                "transposed_table": transposed_table,
                "product_table": product_table,
                "inverse_table": inverse_table,
                "v_table_from_inverse": v_table_from_inverse,
                "residual": residual,
                "adjusted_heights": adjusted_heights
            })
        
        return jsonify(results), 200
        
    except Exception as e:
        # Log the error (in a production environment you'd use a proper logger)
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
