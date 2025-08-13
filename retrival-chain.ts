import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';

import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';

import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

import dotenv from 'dotenv';
dotenv.config();

const model = new ChatGoogleGenerativeAI({
  model: 'gemini-2.5-flash-lite',
  maxOutputTokens: 2048,
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
  Answer the user's question. 
  Context: {context}.
  Question: {input}
`);


// const chain = prompt.pipe(model);
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});

const loader = new CheerioWebBaseLoader("https://js.langchain.com/docs/concepts/lcel");
const docs = await loader.load();
// console.log(docs);
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
});
const splitDocs = await splitter.splitDocuments(docs);

const embeddings = new GoogleGenerativeAIEmbeddings();

const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

//Retrieve relevant documents
const retriever = vectorStore.asRetriever({
  k: 3,
});

const retrivalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever,
});

const response = await retrivalChain.invoke({ input: 'What is LECL?'});
console.log(response);
