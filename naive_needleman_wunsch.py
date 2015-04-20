#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project: Implement the Needleman-Wunsch algorithm for pair-wise alignments.
Purpose: This implementation demonstrates how to align two nucleotide/protein
sequences using the Needleman-Wunsch algorithm developed by Saul B. Needleman
and Christian D. Wunsch.

https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm

A sequence is represented by a string of single letters. A similarity matrix
is dynamically generated based on the input sequences with its match/mismatch
values set to user defined values (or a default match: +1, mismatch: -1,
gap penalty: 0). Alignments are based on maximizing the similarity score.
When multiple maximum values/scores are encountered, the first of the values
is chosen arbitrarily.

A global optimal score is calculated from the match, mismatch and gap penalty
values. Once computed, a traceback is performed. Each step in the back-trace
is determined by the maximum score between cells to the left, to the right
and diagonal from the cell with the previous maximum score.

Note that because this is an exponential problem, only a single
solution is provided.

Numpy arrays are used to improve performance and list accessing. The argparse
module provides parsing for command-line arguments.

This script can be run via the command-line or as a module.
It can be run without input or flags to calculate an alignment using
sample data and the default match/mismatch/gap penalty settings.
It can also be run with input sequences but using default match/mismatch
/gap_penalty settings.

To input sequences to be analyzed:
	-  Supply two strings each containing a sequence, or pass in a file
	   containing the two sequences.
	-  Match, Mismatch and gap penalties can be specified using flags.
	-  An output file can also be specified. If none, results are printed
	   out to std-out.

Note: If sequences inputted via file, sequences must be newline separated
so that the file's first line corresponds to the first sequence and so on.

Example usage:
From the command-line:
> 'python /path/to/naive_sequence_alignment.py
--miss -1 --match 1'
> 'python /path/to/naive_sequence_alignment.py
"""

from datetime import datetime
import numpy as np
import argparse


def parseArgs():
	"""
	Make sense of args specified on the command-line. Returns an argparse object.
	"""
	parser = argparse.ArgumentParser()

	# Define Positional arguments
	parser.add_argument('first_Sequence', nargs='?',
						help="input the First Sequence")
	parser.add_argument('second_Sequence', nargs='?',
						help="input the Second Sequence")
	# Define Optional arguments
	parser.add_argument('-i', '--infile', help="enter input file")
	parser.add_argument('-o', '--outfile', help="enter output file")
	parser.add_argument('--match', help="enter score for matches",
						default=1, type=float)
	parser.add_argument('--miss', help="enter score for mismatches",
						default=-1, type=float)

	args = parser.parse_args()
	return args


class NaiveNeedlemanWunsch(object):
	""" Naive implementation of the Needleman-Wunsch algorithm that constructs
	a similarity matrix, scoring matrix and optimal alignment of the specified
	sequences.
	"""

	def __init__(self, matchScore, mismatchScore, sequence_A, sequence_B,
					outfile):
		self.sequenceA = sequence_A
		self.sequenceB = sequence_B
		self.outfile = outfile
		self.seqA_traceback = []
		self.seqB_traceback = []
		self.size_x = len(sequence_A)
		self.size_y = len(sequence_B)
		self.matchScore = matchScore
		self.mismatchScore = mismatchScore
		self.gap_penalty = 0

		self.init_scoringmatrix()

	def __str__(self):
		return ("module: NaiveNeedlemanWunsch; match: %d; mismatch: %d; gap_penalty: %d;"
				% (self.matchScore, self.mismatchScore, self.gap_penalty))

	def init_scoringmatrix(self):
		""" Creates a zero matrix using numpy's array structure. Inserts gap
		penalties for first row and column.
		"""

		# This creates size_y+1 rows, size_x +1 columns
		self.scoring_matrix = np.zeros((self.size_y+1,self.size_x+1))

		for i in xrange(len(self.scoring_matrix[0,1:])):
			self.scoring_matrix[0,i+1] = self.gap_penalty*(i+1)

		for i in xrange(len(self.scoring_matrix[1:,0])):
			self.scoring_matrix[i+1, 0] = self.gap_penalty*(i+1)

		self.fill_similaritymatrix()
		self.fill_scoringmatrix()

	def bool2score(self, x):
		""" Returns the match/mismatch scores for the boolean values of a
		match/mismatch between SequenceA substring and SequenceB substring.
		"""

		if x:
			return self.matchScore
		else:
			return self.mismatchScore

	def fill_similaritymatrix(self):
		""" Builds a similarity matrix to indicate match/mismatch between
		each element of Sequence A and each element of Sequence B.
		"""

		self.similarity_matrix = np.array([[ self.bool2score(x == y)
										for y in self.sequenceA] for x in self.sequenceB])

	def get_similarity_score(self,x,y):
		""" Given a row and column index,returns the corresponding value from
		the similarity matrix.
		"""

		return self.similarity_matrix[x,y]

	def alignment(self, tracebackpath, original_seq):
		""" Return the new sequence alignment. """

		idx = 0
		new_sequence = []
		# reverse the traceback-path so that we build alignment from the front
		tracebackpath.reverse()

		for i in tracebackpath:
			if i == 'g':
				new_sequence.append('-')
			else:
				new_sequence.append(original_seq[idx])
				idx = idx + 1

		new_sequence = ''.join(new_sequence)
		return new_sequence


	def plot_alignment(self):
		""" Turn results of the traceback part of the Needleman-Wunsch algorithm
		into its corresponding sequence alignment.
		"""

		self.seqA_alignment = self.alignment(self.seqA_traceback, self.sequenceA)
		self.seqB_alignment = self.alignment(self.seqB_traceback, self.sequenceB)


	def traceback_optimalscore(self, currentmax):
		""" Traceback portion of the Needleman-Wunsch algorithm. Pointers (k =
		keep substring, g = add gap) are appended to a list with each step to
		indicate the journey along the matrix.
		"""

		if (currentmax == (0,0,)):
			self.plot_alignment()
		else:
			rowVal = currentmax[0]
			colVal = currentmax[1]
			localmax = self.scoring_matrix[rowVal,colVal]

			if ((self.scoring_matrix[rowVal-1, colVal] + self.gap_penalty) == localmax):
				self.seqA_traceback.append('g')
				self.seqB_traceback.append('k')
				self.traceback_optimalscore((rowVal-1, colVal,))
			elif ((self.scoring_matrix[rowVal, colVal-1] + self.gap_penalty) == localmax):
				self.seqA_traceback.append('k')
				self.seqB_traceback.append('g')
				self.traceback_optimalscore((rowVal, colVal-1,))
			else:
				self.seqA_traceback.append('k')
				self.seqB_traceback.append('k')
				self.traceback_optimalscore((rowVal-1, colVal-1,))

	def fill_scoringmatrix(self):
		""" Scoring Matrix portion of Needleman-Wunsch algorithm.
		Assigns a score to each element of a len(sequence B) + 1 by
		len(sequence A) + 1 matrix, where score is the maximum value
		between the score of cells diagonal, above and to the left
		of the current cell.
		"""

		traceback_Pointer = []
		for (x,y), val in np.ndenumerate(self.scoring_matrix[1:,1:]):
			diagonalscore = self.scoring_matrix[x,y]
			topscore = self.scoring_matrix[x,y+1]
			leftscore = self.scoring_matrix[x+1,y]

			# Set score as the maximizing value
			self.scoring_matrix[x+1,y+1] = max((topscore + self.gap_penalty),
												(leftscore + self.gap_penalty),
											(diagonalscore + self.get_similarity_score(x,y)))

		# Calculate the maximum alignment score
		self.max_alignment_score = max(self.scoring_matrix[self.size_y,:])

		# Find the column location of the maximum alignment score.
		# Row location is last row of matrix
		self.max_col = np.where(self.scoring_matrix[self.size_y,:] == self.max_alignment_score)
		self.max_location = (self.size_y, self.max_col[0][-1])

		# Traceback path of Optimal Score
		self.traceback_optimalscore(self.max_location)

	def return_optimal(self):
		""" Returns a tuple with maximum alignment Score, alignment of Sequence A,
		and alignment of Sequence B
		"""
		return (self.max_alignment_score, self.seqA_alignment, self.seqB_alignment)

	def pretify_return_optimal(self):
		""" Return prettified results of Needleman-Wunsch calculation for printing
		and writing to file.
		"""

		results = self.return_optimal()

		if (self.outfile):
			with open(self.outfile, 'w+') as f:
				f.write('Naive Needleman-Wunsch Results')
				f.write('Ran on: %s ' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
				f.write('Max Alignment Score: %d ' % results[0])
				f.write('%s\t (First Sequence)' % results[1])
				f.write('%s\t (Second Sequence)' % results[2])
		else:
			print ('================')
			print ('RESULTS:')
			print ('Max Alignment Score: %d' % results[0])
			print ('Here\'s one of the possible alignments: ')
			print ('\t%s\t (First Sequence)' % results[1])
			print ('\t%s\t (Second Sequence)' % results[2])
			print ('================')


def main():
	args = parseArgs()

	if (args.infile):
		# Open specified file and parse into individual sequences
		with open(args.infile, 'r+') as f:
			sequences = f.readlines()

		sequences = [i.rstrip('\n') for i in sequences]
		alignSequence = NaiveNeedlemanWunsch(args.match, args.miss, sequences[0],
												sequences[1], args.outfile)

	elif (args.first_Sequence and args.second_Sequence):
		alignSequence = NaiveNeedlemanWunsch(args.match, args.miss, args.first_Sequence,
												args.second_Sequence,args.outfile)

	else:
		# Use example data ATCAGAGTC TTCAGTC
		examples = ['ATCAGAGTC', 'TTCAGTC']
		print ('[No Input Data]')
		print ('Let\'s try out an example using "%s" and "%s"\n' % (examples[0], examples[1]))
		alignSequence = NaiveNeedlemanWunsch(args.match, args.miss,examples[0], examples[1],
												args.outfile)

	# Shows results
	alignSequence.pretify_return_optimal()


if __name__ == "__main__":
	main()