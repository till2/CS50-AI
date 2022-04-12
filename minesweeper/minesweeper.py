import itertools
from queue import Empty
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # No cells in the set -> No mines exist.
        if len(self.cells) == 0:
            return set()
        
        # All cells are mines if the count is equal
        # to the number of cells in the set.
        if len(self.cells) == self.count:
            return self.cells.copy()
        
        # It cannot be the case that the count is greater
        # than the number of cells in the set.
        if len(self.cells) < self.count:
            raise Exception("Impossible Sentence: Count cannot be greater than the number of cells in the set.")
        
        # otherwise
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        # No cells in the set -> No safe cells exist.
        if len(self.cells) == 0:
            return set()
        
        # All cells are save if count is 0.
        if self.count == 0:
            return self.cells.copy()
        
        # otherwise
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1


    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # mark the cell as one of the moves made
        self.moves_made.add(cell)

        # mark the cell as safe
        self.mark_safe(cell)
        
        # add a sentence to the kb
        neighbors = []
        x, y = cell
        count_minus_mines = count
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                neighbor = (i, j)
                if 0 <= i < self.height and 0 <= j < self.width:
                    if neighbor in self.mines:
                        count_minus_mines -= 1
                    elif neighbor in self.safes:
                        pass
                    else:
                        # add undetermined cell
                        neighbors.append(neighbor)

        new_sentence = Sentence(neighbors, count_minus_mines)
        self.knowledge.append(new_sentence)

        found_new_knowledge = True
        while(found_new_knowledge):
            found_new_knowledge = False # set True later if new knowledge was added/interferered

            # mark additional cells as safe or as mines
            for sentence in self.knowledge:
                for cell in sentence.known_mines():
                    if cell not in self.mines:
                        self.mark_mine(cell)

            for cell in sentence.known_safes():
                if cell not in self.safes:
                    self.mark_safe(cell)

            # add any infered new sentences to the AI's knowledge base
            if len(new_sentence.cells):
                for sentence in self.knowledge:
                    if len(sentence.cells):
                        if new_sentence.cells.issubset(sentence.cells) and not new_sentence.__eq__(sentence):
                            # interference = sentence - new_sentence
                            interference_list = list(sentence.cells.difference(new_sentence.cells))
                            interference_count = sentence.count - new_sentence.count
                            interference_sentence = Sentence(interference_list, interference_count)
                            if interference_sentence not in self.knowledge:
                                self.knowledge.append(interference_sentence)
                                found_new_knowledge = True

                        elif sentence.cells.issubset(new_sentence.cells) and not new_sentence.__eq__(sentence):
                            # interference = new_sentence - sentence  
                            interference_list = list(new_sentence.cells.difference(sentence.cells))
                            interference_count = new_sentence.count - sentence.count
                            interference_sentence = Sentence(interference_list, interference_count)
                            if interference_sentence not in self.knowledge:
                                self.knowledge.append(interference_sentence)
                                found_new_knowledge = True
                            

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        moves = []
        for i in range(self.height):
            for j in range(self.width):
                move = (i, j)
                if move in self.safes and move not in self.moves_made:
                    moves.append(move)
        
        if not len(moves):
            return None

        selected_move = random.choice(moves)
        return selected_move
        

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        moves = []
        for i in range(self.height):
            for j in range(self.width):
                move = (i, j)
                if move not in self.mines and move not in self.moves_made:
                    moves.append(move)

        if not len(moves):
            return None

        selected_move = random.choice(moves)
        return selected_move