import {
  CommaSeparatedListOutputParser,
  StringOutputParser,
} from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { StructuredOutputParser } from 'langchain/output_parsers';
import z from 'zod';

import dotenv from 'dotenv';
dotenv.config();

const model = new ChatGoogleGenerativeAI({
  model: 'gemini-2.5-flash',
  maxOutputTokens: 2048,
  apiKey: process.env.GOOGLE_API_KEY,
});

async function callStringOutputParser() {
  const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'Generate a joke based on a word provided by the user'],
    ['human', '{input}'],
  ]);

  //create parser
  const parser = new StringOutputParser();

  //create chain
  const chain = prompt.pipe(model).pipe(parser);

  //call chain
  const response = await chain.invoke({ input: 'dog' });
  return response;
}

async function callListOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Provide 5 synonyms for the word "{input}". Your response should be a single line of comma-separated values.
  `);

  //create parser
  const outputParser = new CommaSeparatedListOutputParser();

  //create chain
  const chain = prompt.pipe(model);

  //call chain
  const response = await chain.invoke({ input: 'success' });
  return response;
}

async function callStructuredOutParser() {
  //create parser

  const prompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following phrase.
    Formatting Instructions: {format_instructions}
    Phrase: {phrase}
    `);
  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: 'the name of the person',
    age: 'the age of the person',
  });

  const chain = prompt.pipe(model).pipe(outputParser);

  const response = await chain.invoke({
    format_instructions: outputParser.getFormatInstructions(),
    phrase: 'John is 30 years old and lives in New York.',
  });
  return response;
}

async function callZodParser(){
  const prompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following phrase.
    Formatting Instructions: {format_instructions}
    Phrase: {phrase}
  `);

  const outputParser = StructuredOutputParser.fromZodSchema(
    z.object({
      recipe: z.string().describe('name of recipe'),
      ingredients: z.array(z.string()).describe('ingredients'),
    })
  );

  const chain = prompt.pipe(model).pipe(outputParser);

  const response = await chain.invoke({
    format_instructions: outputParser.getFormatInstructions(),
    phrase: 'Pizza is a popular dish made with a round, flat base of leavened wheat-based dough topped with tomatoes, cheese, and often various other ingredients (such as anchovies, olives, vegetables, meat, etc.), baked at a high temperature, traditionally in a wood-fired oven.',
  });
  return response;
}

// const response = await callListOutputParser();
// const response = await callListOutputParser();
// const response = await callStructuredOutParser();
const response = await callZodParser();

console.log(response);
