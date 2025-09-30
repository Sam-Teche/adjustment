from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re

app = Flask(__name__)
CORS(app)


class AdjustmentCalculator:
    """Handles all adjustment calculations with frontend-compatible output"""
    
    def __init__(self, df, benchmarks, formulas):
        self.df = df
        self.benchmarks = benchmarks
        self.formulas = formulas
        self.n = len(df)
        
    def create_distance_matrix(self):
        """Create diagonal distance matrix"""
        return np.diag(self.df['Distances_Km'].values)
    
    @staticmethod
    def modify_equation(formula):
        """Convert formula to standard format with V variables"""
        parts = formula.replace('-', ' -').replace('+', ' +').split()
        result = []
        
        for part in parts:
            if part in ['+', '-']:
                continue
                
            operator = ''
            if part.startswith(('+', '-')):
                operator = part[0]
                part = part[1:]
            
            if operator:
                result.append(operator)
            
            if re.match(r'[hH]\d+', part):
                var_index = re.match(r'[hH](\d+)', part).group(1)
                result.append(f'V{var_index} + {part}')
            elif '=' in part:
                left, right = part.split('=', 1)
                if re.match(r'[hH]\d+', left):
                    var_index = re.match(r'[hH](\d+)', left).group(1)
                    result.append(f'V{var_index} + {left}={right}')
                else:
                    result.append(part)
            else:
                result.append(part)
        
        return ' '.join(result)
    
    @staticmethod
    def extract_coefficients(equation, num_vars):
        """Extract coefficient vector from equation"""
        coeffs = {}
        matches = re.finditer(r'([+-]?)\s*V(\d+)', equation)
        
        for match in matches:
            sign = match.group(1) or '+'
            var_index = int(match.group(2))
            coeffs[var_index] = sign + '1'
        
        return coeffs
    
    def build_coefficient_table(self):
        """Build coefficient table for frontend compatibility"""
        table = []
        modified_eqns = []
        
        for i, formula in enumerate(self.formulas):
            modified = self.modify_equation(formula)
            modified_eqns.append(modified)
            coeffs = self.extract_coefficients(modified, self.n)
            row = [f"Eqn {i+1}"] + [coeffs.get(j, '0') for j in range(1, self.n + 1)]
            table.append(row)
        
        return table, modified_eqns
    
    def calculate_w_values(self):
        """Calculate misclosure vector w"""
        w_values = []
        
        for formula in self.formulas:
            formula = formula.replace('hh', 'h')
            parts = formula.split('=')
            
            if len(parts) != 2:
                continue
            
            left_val = self._evaluate_side(parts[0])
            right_val = self._evaluate_side(parts[1])
            
            w_values.append(-(right_val - left_val))
        
        return w_values
    
    def _evaluate_side(self, expression):
        """Evaluate one side of equation"""
        value = 0
        terms = re.findall(r'([+-]?)([hHbB])(\d+)', expression)
        
        for sign, var_type, idx in terms:
            multiplier = -1 if sign == '-' else 1
            idx = int(idx) - 1
            
            if var_type.lower() == 'h' and idx < len(self.df):
                value += multiplier * self.df['h_obs'].iloc[idx]
            elif var_type.upper() == 'B' and idx < len(self.benchmarks):
                value += multiplier * self.benchmarks[idx]
        
        if not terms and expression.strip():
            try:
                value = eval(expression.strip())
            except:
                pass
        
        return value
    
    def format_value(self, val, decimals=3):
        """Format value with + prefix for positive numbers"""
        if abs(val) < 1e-10:
            return "0"
        formatted = f"{val:.{decimals}f}"
        return f"+{formatted}" if val > 0 else formatted
    
    def multiply_coefficients_with_distances(self, coeff_table, distance_matrix):
        """Multiply coefficients with distances"""
        result_table = []
        
        for row in coeff_table:
            eqn_name = row[0]
            coeffs = [int(c.replace('+', '')) if c != '0' else 0 for c in row[1:]]
            
            result_row = [eqn_name]
            for i, coeff in enumerate(coeffs):
                if coeff != 0:
                    weighted = coeff * distance_matrix[i, i]
                    result_row.append(self.format_value(weighted))
                else:
                    result_row.append("0")
            
            result_table.append(result_row)
        
        return result_table
    
    def transpose_coefficient_table(self, coeff_table):
        """Transpose coefficient table"""
        eqn_names = [row[0] for row in coeff_table]
        num_cols = len(coeff_table[0]) - 1
        
        transposed = [["Parameter"] + eqn_names]
        for j in range(1, num_cols + 1):
            row = [f"V{j}"] + [coeff_table[i][j] for i in range(len(coeff_table))]
            transposed.append(row)
        
        return transposed
    
    def calculate_product_table(self, result_table):
        """Calculate product table (A*P*A^T)"""
        eqn_names = [row[0] for row in result_table]
        
        # Extract numeric matrix (exclude equation names and w column)
        numeric_matrix = []
        for row in result_table:
            values = [float(val.replace('+', '')) for val in row[1:-1]]
            numeric_matrix.append(values)
        
        matrix = np.array(numeric_matrix)
        product = matrix @ matrix.T
        
        # Format as table
        product_table = []
        for i, row in enumerate(product):
            formatted_row = [eqn_names[i]] + [self.format_value(val, 4) for val in row]
            product_table.append(formatted_row)
        
        return product_table
    
    def calculate_inverse_table(self, product_table):
        """Calculate inverse of product table"""
        eqn_names = [row[0] for row in product_table]
        
        # Extract numeric matrix
        numeric_matrix = np.array([
            [float(val.replace('+', '')) for val in row[1:]]
            for row in product_table
        ])
        
        try:
            inverse = np.linalg.inv(numeric_matrix)
            
            inverse_table = []
            for i, row in enumerate(inverse):
                formatted_row = [eqn_names[i]] + [self.format_value(val, 4) for val in row]
                inverse_table.append(formatted_row)
            
            return inverse_table
        except np.linalg.LinAlgError:
            return None
    
    def calculate_k_values(self, inverse_table, w_values):
        """Calculate K values (Lagrange multipliers)"""
        if inverse_table is None:
            return None
        
        # Extract numeric inverse matrix
        inverse_matrix = np.array([
            [float(val.replace('+', '')) for val in row[1:]]
            for row in inverse_table
        ])
        
        # Calculate k = -N^(-1) * w
        k = -inverse_matrix @ np.array(w_values)
        
        return [[f"K{i+1}", self.format_value(val, 4)] for i, val in enumerate(k)]
    
    def calculate_residuals(self, distance_matrix, coeff_table, k_values):
        """Calculate residuals"""
        # Extract coefficient matrix
        A = np.array([
            [int(c.replace('+', '')) if c != '0' else 0 for c in row[1:]]
            for row in coeff_table
        ])
        
        # Extract k values
        k = np.array([float(row[1].replace('+', '')) for row in k_values])
        
        # Calculate v = -P * A^T * k
        v = -distance_matrix @ A.T @ k
        
        return [[f"Residual {i+1}", self.format_value(val, 4)] for i, val in enumerate(v)]
    
    def calculate_adjusted_heights(self, residuals):
        """Calculate adjusted heights"""
        residual_values = [float(row[1].replace('+', '')) for row in residuals]
        
        adjusted = []
        for i, (h_obs, v) in enumerate(zip(self.df['h_obs'], residual_values)):
            adjusted_height = h_obs + v
            adjusted.append([f"Adjusted Height {i+1}", f"{adjusted_height:.4f}"])
        
        return adjusted
    
    def perform_full_adjustment(self):
        """Perform complete adjustment and return all results"""
        # Build coefficient table
        coeff_table, modified_eqns = self.build_coefficient_table()
        
        # Create distance matrix
        P = self.create_distance_matrix()
        
        # Calculate w values
        w_values = self.calculate_w_values()
        
        # Create weighted coefficient table
        result_table = self.multiply_coefficients_with_distances(coeff_table, P)
        
        # Add w values to result table
        for i, row in enumerate(result_table):
            row.append(self.format_value(w_values[i], 3))
        
        # Transpose table
        transposed_table = self.transpose_coefficient_table(coeff_table)
        
        # Calculate product table
        product_table = self.calculate_product_table(result_table)
        
        # Calculate inverse
        inverse_table = self.calculate_inverse_table(product_table)
        
        # Calculate K values
        k_values = self.calculate_k_values(inverse_table, w_values)
        
        # Calculate residuals
        residuals = None
        adjusted_heights = None
        
        if k_values:
            residuals = self.calculate_residuals(P, coeff_table, k_values)
            if residuals:
                adjusted_heights = self.calculate_adjusted_heights(residuals)
        
        return {
            'modified_equations': modified_eqns,
            'coefficient_table': coeff_table,
            'result_table': result_table,
            'transposed_table': transposed_table,
            'product_table': product_table,
            'inverse_table': inverse_table,
            'v_table_from_inverse': k_values,
            'residual': residuals,
            'adjusted_heights': adjusted_heights,
            'distance_matrix': P
        }


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route('/api/adjustment', methods=['POST'])
def perform_adjustment():
    try:
        data = request.get_json()
        
        # Validate input
        required = ['benchmarks', 'observations', 'num_parameters', 'formulas']
        if missing := [f for f in required if f not in data]:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        
        # Create DataFrame
        df = pd.DataFrame(data['observations'])
        
        # Validate columns
        required_cols = ['Obs_No', 'Distances_Km', 'h_obs']
        if missing_cols := [c for c in required_cols if c not in df.columns]:
            return jsonify({"error": f"Missing columns: {', '.join(missing_cols)}"}), 400
        
        df['Distances_Km'] = df['Distances_Km'].astype(float)
        df['h_obs'] = df['h_obs'].astype(float)
        
        n = len(df)
        m = data['num_parameters']
        r = n - m
        
        # Validate condition equations
        if r <= 0 and data['formulas']:
            return jsonify({"error": "No condition equations needed but formulas provided"}), 400
        if r > 0 and len(data['formulas']) != r:
            return jsonify({"error": f"Expected {r} condition equations but got {len(data['formulas'])}"}), 400
        
        # Prepare base results
        results = {
            "num_observations": n,
            "num_parameters": m,
            "num_condition_equations": r,
            "observation_data": df.to_dict(orient='records'),
            "benchmark_data": [{"Point": f"B{i+1}", "Elevation": b} 
                             for i, b in enumerate(data['benchmarks'])]
        }
        
        # Perform adjustment if needed
        if r > 0:
            calc = AdjustmentCalculator(df, data['benchmarks'], data['formulas'])
            adj_results = calc.perform_full_adjustment()
            
            if adj_results['v_table_from_inverse'] is None:
                return jsonify({"error": "Matrix is singular - cannot compute adjustment"}), 400
            
            # Add all adjustment results
            results.update(adj_results)
            results['distance_matrix'] = adj_results['distance_matrix'].tolist()
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)