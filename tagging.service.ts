// tagging.service.ts
import { Injectable } from '@nestjs/common';
import axios from 'axios';
import * as fs from 'fs';
import * as FormData from 'form-data';

@Injectable()
export class TaggingService {
  async predictCategory(filePath: string, title: string, description: string) {
    const form = new FormData();
    form.append('image', fs.createReadStream(filePath));
    form.append('title', title);
    form.append('description', description);

    const res = await axios.post('http://localhost:8000/predict', form, {
      headers: form.getHeaders(),
    });

    return res.data;
  }
}
