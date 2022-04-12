import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob = 1

    # 0 genes
    for person in people:
        if person not in one_gene and person not in two_genes:

            if not people[person]["mother"]:
                # no parents
                joint_prob *= PROBS["gene"][0]

            else:
                # has parents

                # prob for no gene from mother:
                # given mother has 1 gene
                if people[person]["mother"] in one_gene:
                    joint_prob *= 0.5

                # given mother has 2 genes
                elif people[person]["mother"] in two_genes:
                    joint_prob *= PROBS["mutation"]

                # given mother has 0 genes
                else:
                    joint_prob *= (1 - PROBS["mutation"])


                # prob for no gene from father:
                # given father has 1 gene
                if people[person]["father"] in one_gene:
                    joint_prob *= 0.5

                # given father has 2 genes
                elif people[person]["father"] in two_genes:
                    joint_prob *= PROBS["mutation"]

                # given father has 0 genes
                else:
                    joint_prob *= (1 - PROBS["mutation"])

            # trait or not given 0 genes
            if person in have_trait:
                joint_prob *= PROBS["trait"][0][True]
            else:
                joint_prob *= PROBS["trait"][0][False]

    # 1 gene
    for person in one_gene:

        if not people[person]["mother"]:
            # no parents
            joint_prob *= PROBS["gene"][1]

        else:
            # has parents

            # prob for 1 gene from mother:
            # given mother has 1 gene
            if people[person]["mother"] in one_gene:
                # prob for one gene from mother and not father
                gene_from_mother_prob = 0.5
                gene_not_from_mother_prob = 0.5

            # given mother has 2 genes
            elif people[person]["mother"] in two_genes:
                gene_from_mother_prob = 1 - PROBS["mutation"]
                gene_not_from_mother_prob = PROBS["mutation"]
            
            # given mother has 0 genes
            else:
                gene_from_mother_prob = PROBS["mutation"]
                gene_not_from_mother_prob = 1 - PROBS["mutation"]


            # prob for 1 gene from father:
            # given father has 1 gene
            if people[person]["father"] in one_gene:
                gene_from_father_prob = 0.5
                gene_not_from_father_prob = 0.5

            # given father has 2 genes
            elif people[person]["father"] in two_genes:
                gene_from_father_prob = 1 - PROBS["mutation"]
                gene_not_from_father_prob = PROBS["mutation"]

            # given father has 0 genes
            else:
                gene_from_father_prob = PROBS["mutation"]
                gene_not_from_father_prob = 1 - PROBS["mutation"]
            
            # calculate prob for 1 gene from mother and 0 from father
            one_gene_from_mother_prob = gene_from_mother_prob * gene_not_from_father_prob
            one_gene_from_father_prob = gene_from_father_prob * gene_not_from_mother_prob

            one_gene_prob = one_gene_from_mother_prob + one_gene_from_father_prob
            joint_prob *= one_gene_prob


        # trait or not given 1 gene
        if person in have_trait:
            joint_prob *= PROBS["trait"][1][True]
        else:
            joint_prob *= PROBS["trait"][1][False]

    # 2 genes
    for person in two_genes:

        if not people[person]["mother"]:
            # no parents
            joint_prob *= PROBS["gene"][2]

        else:
            # has parents

            # prob for gene from mother:
            # given mother has 1 gene
            if people[person]["mother"] in one_gene:
                joint_prob *= 0.5

            # given mother has 2 genes
            elif people[person]["mother"] in two_genes:
                joint_prob *= (1 - PROBS["mutation"])

            # given mother has 0 genes
            else:
                joint_prob *= PROBS["mutation"]


            # prob for gene from father:
            # given father has 1 gene
            if people[person]["father"] in one_gene:
                joint_prob *= 0.5

            # given father has 2 genes
            elif people[person]["father"] in two_genes:
                joint_prob *= (1 - PROBS["mutation"])

            # given father has 0 genes
            else:
                joint_prob *= PROBS["mutation"]

        # trait or not given 2 genes
        if person in have_trait:
            joint_prob *= PROBS["trait"][2][True]
        else:
            joint_prob *= PROBS["trait"][2][False]

    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    persons = list(probabilities.keys())

    for person in persons:

        # update probability distribution for gene
        # 1 gene
        if person in one_gene:
            probabilities[person]["gene"][1] += p

        # 2 genes
        elif person in two_genes:
            probabilities[person]["gene"][2] += p

        # 0 genes
        else:
            probabilities[person]["gene"][0] += p


        # update probability distribution for trait
        # trait
        if person in have_trait:
            probabilities[person]["trait"][True] += p

        # no trait
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    persons = list(probabilities.keys())

    for person in persons:

        # normalize probability distribution for gene 
        genes_sum = 0
        for num_genes in range(3):
            genes_sum += probabilities[person]["gene"][num_genes]
        
        alpha = 1 / genes_sum
        
        for num_genes in range(3):
            probabilities[person]["gene"][num_genes] *= alpha
        
        # normalize probability distribution for trait
        trait_sum = probabilities[person]["trait"][True] + \
                    probabilities[person]["trait"][False]
        
        alpha = 1 / trait_sum

        probabilities[person]["trait"][True] *= alpha
        probabilities[person]["trait"][False] *= alpha


if __name__ == "__main__":
    main()
