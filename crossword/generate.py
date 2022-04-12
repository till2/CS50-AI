import sys
from tkinter import YView

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        variables = list(self.domains.keys())

        for var in variables:
            unconsistent_values = list(filter(lambda x: len(x) != var.length, self.domains[var]))
            for value in unconsistent_values:
                self.domains[var].remove(value)


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        revised = False
        overlap = self.crossword.overlaps[x,y]
        removed_x_values = []

        for x_val in self.domains[x]:

            found_corresponding_y = False

            for y_val in self.domains[y]:

                if overlap:
                    if x_val[overlap[0]] == y_val[overlap[1]]:
                        found_corresponding_y = True
                        break
                else:
                    found_corresponding_y = True
                    break
            
            if not found_corresponding_y:
                revised = True
                removed_x_values.append(x_val)

        # remove x-values with no corresponding y-value
        for x_val in removed_x_values:
            self.domains[x].remove(x_val)
        
        return revised


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        # if arcs is None -> enqueue all arcs
        if arcs is None:
            arcs = []
            for x in self.domains.keys():
                for y in self.domains.keys():
                    if x is not y:
                        arcs.append((x,y))
                        arcs.append((y,x))
        
        while arcs:
            arc = arcs.pop(0)
            x, y = arc
            if self.revise(x, y):

                if not len(self.domains[x]):
                    return False
                
                neighbors = self.crossword.neighbors(x)
                neighbors.remove(y)

                for z in neighbors:
                    arcs.append((z,x))
        
        return True
        

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if assignment:
            for var in assignment.keys():
                if len(assignment[var]) != 1:
                    return False
        return True


    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        if assignment:

            # ensure distinct values
            val_list = list(assignment.values())
            val_set = set(val_list)

            if len(val_list) != len(val_set):
                return False

            # ensure correct length
            for var, val in assignment.items():
                if not var.length == len(val):
                    return False

            # no conflicts between neighboring vars
            conflicts = False

            for x, xval in assignment.items():

                neighbors = self.crossword.neighbors(x)

                for y in neighbors:
                    if y in assignment.keys():
                        yval = assignment[y]
                        overlap = self.crossword.overlaps[x,y]
                        
                        xpos, ypos = overlap

                        if xval[xpos] != yval[ypos]:
                            return False

        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        values = list(self.domains[var])
        neighbors = self.crossword.neighbors(var)
        restrictions = dict()

        # sort values according to the least-constraining values heuristic

        for val in values:
            restrictions[val] = 0

            for n in neighbors:
                if n not in assignment.keys():

                    overlap = self.crossword.overlaps[var, n]
                    if overlap:
                        var_pos, n_pos = overlap

                        for n_val in self.domains[n]:
                            if val[var_pos] != n_val[n_pos]:
                                restrictions[val] += 1
        
        sorted_values = [val for val, _ in sorted(restrictions.items(), key=lambda item: item[1])]   

        return sorted_values


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """

        # list of vars that don't have a value assigned yet
        unassigned_vars = [var for var in self.domains.keys() if var not in assignment.keys()]

        remaining_values = {var:len(self.domains[var]) for var in unassigned_vars}

        degrees = {var:len(self.crossword.neighbors(var)) for var in unassigned_vars}

        # sort by:
        # first key: fewest number of remaining values in its domain (ascending)
        # second key: degree/ neighbors-count (descending -> same as ascending negated degrees)
        sorted_unassigned_vars = sorted(
            unassigned_vars, 
            key=lambda var: (remaining_values[var], -degrees[var]))

        selected_var = sorted_unassigned_vars[0]
        return selected_var


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        assigned_vars = list(assignment.keys())
        all_vars = list(self.domains.keys())

        # return if assignment complete
        if len(assigned_vars) == len(all_vars):

            # check if assignment is satisfactory
            if self.consistent(assignment):
                return assignment
            else:
                return None
        
        var = self.select_unassigned_variable(assignment)

        for val in self.order_domain_values(var, assignment):

            new_assignment = assignment.copy()
            new_assignment[var] = val

            if self.consistent(new_assignment):

                assignment[var] = val
                result = self.backtrack(assignment)
                
                if result is not None:
                    return result
                
                del assignment[var]
        
        # failure
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
