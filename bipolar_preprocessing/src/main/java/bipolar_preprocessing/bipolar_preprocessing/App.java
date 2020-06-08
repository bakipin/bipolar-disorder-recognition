package bipolar_preprocessing.bipolar_preprocessing;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.util.List;
import java.util.StringTokenizer;

import org.antlr.v4.runtime.Token;

import zemberek.morphology.TurkishMorphology;
import zemberek.morphology.analysis.WordAnalysis;
import zemberek.tokenization.TurkishSentenceExtractor;
import zemberek.tokenization.TurkishTokenizer;
import zemberek.tokenization.antlr.TurkishLexer;


public class App 
{
	public static void main( String[] args ) throws IOException
	{
		TurkishTokenizer tokenizer = TurkishTokenizer
				.builder()
				.ignoreTypes(TurkishLexer.NewLine, TurkishLexer.SpaceTab, TurkishLexer.Time, TurkishLexer.Punctuation)
				.build();
		TurkishMorphology morphology = TurkishMorphology.createWithDefaults();

		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream("transcripts.csv")));
		PrintWriter writer = new PrintWriter(new File("transcripts_preprocessed.csv"));
		String strLine = "";
		StringTokenizer st = null;

		int lineNumber = 0, tokenNumber = 0;;
		StringBuilder sb = new StringBuilder();

		while ((strLine = reader.readLine()) != null) {
			System.out.println(strLine.replaceAll("\"", ""));
			System.out.println("aaaaaa");
			if(lineNumber == 0) {
				sb.append("file_name,transcript,num_verb,num_noun,num_adj,num_adv,num_neg,num_narr,num_past,num_prog,num_fut,num_pres,a1sg,a2sg,a3sg,a1pl,a2pl,a3pl");
				sb.append("\n");
				lineNumber+=1;
			}
			strLine = strLine.replaceAll("\"", "");
			//break comma separated line using ","
			st = new StringTokenizer(strLine, ",");
			int narr = 0, past = 0, prog = 0, fut = 0, pres = 0;
			int neg = 0;
			int verb = 0, noun = 0, adj = 0, adv = 0;
			int a1sg = 0, a2sg = 0, a3sg = 0, a1pl = 0, a2pl = 0, a3pl = 0;
			while (st.hasMoreTokens()) {
				sb.append(st.nextToken());
				sb.append(',');
				String str = st.nextToken();
				//System.out.println(str);
				List<Token> tokens = tokenizer.tokenize(str);
				String sent = "";
				for(Token tok : tokens) {
					WordAnalysis results = morphology.analyze(tok);

					if(results.getAnalysisResults().isEmpty()) {
						sent += tok.getText() + " ";
						continue;
					}

					sent += results.getAnalysisResults().get(0).getLemmas().get(0)+ " ";
					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Verb") &&
							!results.getAnalysisResults().get(0).getMorphemes().toString().contains("Noun") &&
							!results.getAnalysisResults().get(0).getMorphemes().toString().contains("Adj") &&
							!results.getAnalysisResults().get(0).getMorphemes().toString().contains("Adv") &&
							!tok.getText().equalsIgnoreCase("bana") && !tok.getText().equalsIgnoreCase("onun")) {
						if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("A1sg")) a1sg+=1;
						if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("A2sg")) a2sg+=1;
						if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("A3sg")) a3sg+=1;
						if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("A1pl")) a1pl+=1;
						if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("A2pl")) a2pl+=1;
						if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("A3pl")) a3pl+=1;
					}
					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Verb")) verb+=1;
					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Noun")) noun+=1;
					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Adj")) adj+=1;
					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Adv")) adv+=1;

					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Neg")) neg+=1;

					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Narr")) narr+=1;
					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Past")) past+=1;
					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Prog1")) prog+=1;
					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Fut")) fut+=1;
					if(results.getAnalysisResults().get(0).getMorphemes().toString().contains("Pres")) pres+=1;
				}
				sb.append(sent);
				sb.append(',');
				sb.append(verb);
				sb.append(',');
				sb.append(noun);
				sb.append(',');
				sb.append(adv);
				sb.append(',');
				sb.append(adj);
				sb.append(',');
				sb.append(neg);
				sb.append(',');
				sb.append(narr);
				sb.append(',');
				sb.append(past);
				sb.append(',');
				sb.append(prog);
				sb.append(',');
				sb.append(fut);
				sb.append(',');
				sb.append(pres);
				sb.append(',');
				sb.append(a1sg);
				sb.append(',');
				sb.append(a2sg);
				sb.append(',');
				sb.append(a3sg);
				sb.append(',');
				sb.append(a1pl);
				sb.append(',');
				sb.append(a2pl);
				sb.append(',');
				sb.append(a3pl);
				sb.append(System.getProperty("line.separator"));

			}
			//reset token number
			tokenNumber = 0;
			System.out.println("----------------------------------------------");

		}
		writer.write(sb.toString());
		writer.close();
	}
}
